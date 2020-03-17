import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import softplus
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm, trange
from sklearn import metrics
import matplotlib.pyplot as plt
# % matplotlib inline
import seaborn as sns
import pandas as pd
import numpy as np
import time
import datetime
import pyro
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.optim import ClippedAdam

from models import GMU
from models import Maxout_MLP
from pyro_models import GMU_Pyro


#Train-Validation-Test
class Training_Testing():

    def __init__(self, Data_train_tensor_text, Data_train_tensor_image, Labels_train_tensor, 
                 Data_test_tensor_text, Data_test_tensor_image, Labels_test_tensor, 
                 Data_val_tensor_text, Data_val_tensor_image, Labels_val_tensor,
                 Label_names = None, hidden_layer_size = 512, num_maxout_units = 2, weight_decay= 0.1, reg_param = 0.1, scheduler_step_size = 30, scheduler_lr_fraction = 0.8,
                 hidden_activation = "tanh", batch_size = 32, epochs = 10, sigmoid_thresh = 0.2, learning_rate = 2e-5, num_labels = 23, dropout = 0.1, max_norm = 5, annealing_factor = 2.4):


      self.gmu = GMU(num_maxout_units = num_maxout_units, hidden_layer_size = hidden_layer_size, hidden_activation = hidden_activation, dropout = dropout).cuda()
      self.mlp = Maxout_MLP(hidden_layer_size, hidden_layer_size, dropout = dropout, num_labels = num_labels, num_maxout_units = num_maxout_units).cuda()
      self.pyro = GMU_Pyro(model = self.gmu)

      #self.inference = SVI(self.pyro.gmu_model, self.pyro.gmu_guide, ClippedAdam({"lr": learning_rate}), loss = self.custom_mc_elbo)
      #self.inference = SVI(self.pyro.gmu_model, self.pyro.gmu_guide, ClippedAdam({"lr": learning_rate}), loss = Trace_ELBO())
      self.inference = SVI(self.pyro.gmu_model, self.pyro.gmu_guide, ClippedAdam({"lr": learning_rate}), loss = self.simple_elbo_kl_annealing)
      
      self.label_names = Label_names
      self.num_labels = num_labels
      self.batch_size = batch_size
      self.learning_rate = learning_rate
      self.max_norm = max_norm
      self.epochs = epochs
      self.sigmoid_thresh = sigmoid_thresh
      self.scheduler_step_size = scheduler_step_size
      self.scheduler_lr_fraction = scheduler_lr_fraction
      self.weight_decay = weight_decay
      self.reg_param = reg_param
      self.optimizer = self.SetOptimizer()
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.results = pd.DataFrame(0, index=['Recall','Precision','F_Score'], columns=['micro', 'macro', 'weighted', 'samples']).astype(float)
      self.epoch_loss_set = []
      self.epoch_gmu_loss_set = []
      self.train_dataloader = self.SetTrainDataloader_MM(Data_train_tensor_text, Data_train_tensor_image, Labels_train_tensor)
      self.test_dataloader = self.SetTestDataloader_MM(Data_test_tensor_text, Data_test_tensor_image, Labels_test_tensor) 
      self.scheduler = self.SetScheduler()
      self.annealing_factor = annealing_factor

      self.val_accuracy_set = [] 
      #self.val_loss_set = [] 
      self.val_dataloader = self.SetValDataloader_MM(Data_val_tensor_text, Data_val_tensor_image, Labels_val_tensor)
      self.class_wise_metrics = None
      self.predictions = None


    # custom Elbo
    def custom_mc_elbo(self, model, guide, *args):
      guide_trace = poutine.trace(guide).get_trace(*args)
      model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(*args)
      elbo = model_trace.log_prob_sum() - guide_trace.log_prob_sum()
      return -elbo


    def simple_elbo_kl_annealing(self, model, guide, *args):
        annealing_factor = self.annealing_factor
        latents_to_anneal = ["my_latent"]
        guide_trace = poutine.trace(guide).get_trace(*args)
        model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(*args)
        elbo = 0.0
        for site in model_trace.nodes.values():
            if site["type"] == "sample":
                factor = annealing_factor if site["name"] in latents_to_anneal else 1.0
                elbo = elbo + factor * site["fn"].log_prob(site["value"]).sum()
        for site in guide_trace.nodes.values():
            if site["type"] == "sample":
                factor = annealing_factor if site["name"] in latents_to_anneal else 1.0
                elbo = elbo - factor * site["fn"].log_prob(site["value"]).sum()
        return -elbo


    def L2_Regularizer(self):
      reg_loss = 0.0
      for param in self.gmu.parameters():
          reg_loss = reg_loss + param.pow(2.0).sum()
      return self.reg_param*reg_loss

    def L1_Regularizer(self):
      reg_loss = 0.0
      for param in self.gmu.parameters():
          reg_loss = reg_loss + torch.sum(torch.abs(param))
      return self.reg_param*reg_loss


    def SetOptimizer(self) :

      optimizer = AdamW(self.mlp.parameters(), lr=self.learning_rate,  eps = 1e-6, weight_decay=self.weight_decay)
      #optimizer = Adam(self.mlp.parameters(), lr=self.learning_rate,  eps = 1e-6, weight_decay=self.weight_decay)
      return(optimizer)

    

    def SetScheduler(self) :

      '''
      scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = 10, 
                                                 num_training_steps = self.epochs*len(self.train_dataloader))
      '''
      scheduler = StepLR(self.optimizer, step_size = self.scheduler_step_size, gamma = self.scheduler_lr_fraction)
      return(scheduler) 



    def Get_Metrics(self, actual, predicted) :


      averages = ('micro', 'macro', 'weighted', 'samples')
      for average in averages:
          precision, recall, fscore, _ = metrics.precision_recall_fscore_support(actual, predicted, average=average)
          self.results[average]['Recall'] += recall
          self.results[average]['Precision'] += precision
          self.results[average]['F_Score'] += fscore


    def Plot_Training_Epoch_Loss(self) :

      sns.set(style='darkgrid')
      sns.set(font_scale=1.5)
      plt.rcParams["figure.figsize"] = (12,6)
      plt.plot(self.epoch_loss_set, 'b-o')
      #plt.ylim(top = 1, bottom = 0)
      plt.title("Training loss")
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.savefig('Training_Epoch_Loss.png',bbox_inches='tight')
      plt.show()


    def Plot_Training_Epoch_SVI_Loss(self) :

      sns.set(style='darkgrid')
      sns.set(font_scale=1.5)
      plt.rcParams["figure.figsize"] = (12,6)
      plt.plot(self.epoch_gmu_loss_set, 'b-o')
      plt.title("Training loss (SVI)")
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.savefig('Training_Epoch_SVI_Loss.png',bbox_inches='tight')
      plt.show()

    
    def Plot_Training_Epoch_Accuracy(self) :

      sns.set(style='darkgrid')
      sns.set(font_scale=1.5)
      plt.rcParams["figure.figsize"] = (12,6)
      plt.plot(self.val_accuracy_set, 'b-o')
      plt.title("Weighted F1 Score")
      plt.xlabel("Epoch")
      plt.ylabel("Validation Accuracy")
      plt.savefig('Training_Validation_Accuracy.png',bbox_inches='tight')
      plt.show()


    def format_time(self, elapsed):
      '''
      Takes a time in seconds and returns a string hh:mm:ss
      '''
      # Round to the nearest second.
      elapsed_rounded = int(round((elapsed)))
      return str(datetime.timedelta(seconds=elapsed_rounded))


    def SetTrainDataloader_MM(self, Data_train_tensor_text, Data_train_tensor_image, Labels_train_tensor) :

      train_dataset = TensorDataset(Data_train_tensor_text, Data_train_tensor_image, Labels_train_tensor)
      train_sampler = RandomSampler(train_dataset)
      train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size = self.batch_size)
      return(train_dataloader)


    def SetTestDataloader_MM(self, Data_test_tensor_text, Data_test_tensor_image, Labels_test_tensor) :
      
      test_dataset = TensorDataset(Data_test_tensor_text, Data_test_tensor_image, Labels_test_tensor)
      test_sampler = SequentialSampler(test_dataset)
      #test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size = self.batch_size)
      test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size = Data_test_tensor_text.shape[0])
      return(test_dataloader)

    
    def SetValDataloader_MM(self, Data_val_tensor_text, Data_val_tensor_image, Labels_val_tensor) :
      
      val_dataset = TensorDataset(Data_val_tensor_text, Data_val_tensor_image, Labels_val_tensor)
      val_sampler = SequentialSampler(val_dataset)
      #test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size = self.batch_size)
      val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size = Data_val_tensor_text.shape[0])
      return(val_dataloader)

   
    def Train(self) :
      
      # clear param store
      pyro.clear_param_store()
      
      for _ in trange(self.epochs, desc="Epoch"):
        
        self.gmu.train()
        self.mlp.train()
        epoch_loss = 0
        epoch_gmu_loss = 0

        # Measure how long the training epoch takes.
        t0 = time.time()
    
        for step_num, batch_data in enumerate(self.train_dataloader):

          # Progress update every 30 batches.
          if step_num % 30 == 0 and not step_num == 0:
            elapsed = self.format_time(time.time() - t0)
            print('  Batch : ',step_num, ' , Time elapsed : ',elapsed)

          samples_text, samples_image, labels = tuple(t.to(self.device) for t in batch_data)
          self.optimizer.zero_grad()

          ##### Pyro - GMU #####
          gmu_loss = self.inference.step(samples_text.float(), samples_image.float(), labels.t()) + self.L2_Regularizer()
          attendout = self.gmu(samples_text, samples_image)
          epoch_gmu_loss += gmu_loss.detach().cpu().numpy()
          #epoch_gmu_loss += gmu_loss

          ##### MLP ####
          logits = self.mlp(attendout)
          loss_fct = BCEWithLogitsLoss()
          batch_loss = loss_fct(logits.view(-1, self.num_labels).float(), labels.view(-1, self.num_labels).float())
          batch_loss.backward()
          clip_grad_norm_(self.mlp.parameters(), norm_type = 2, max_norm = self.max_norm)
          self.optimizer.step()
          self.scheduler.step()
          epoch_loss += batch_loss.item()

        avg_epoch_loss = epoch_loss/len(self.train_dataloader)
        avg_epoch_gmu_loss = epoch_gmu_loss/len(self.train_dataloader)
        print("\nTrain loss for epoch: ",avg_epoch_loss)
        print("\nTrain loss for epoch (gmu): ",avg_epoch_gmu_loss)
        print("\nTraining epoch took: {:}".format(self.format_time(time.time() - t0)))
        self.epoch_loss_set.append(avg_epoch_loss)
        self.epoch_gmu_loss_set.append(avg_epoch_gmu_loss)


        '''
        #Validation loss on the epoch
        epoch_val_loss = 0

        for batch_data in self.val_dataloader:
          samples_text, samples_image, labels = tuple(t.to(self.device) for t in batch_data)
          with torch.no_grad():
            attendout = self.gmu(samples_text.float(), samples_image.float())
            logits = self.mlp(attendout)
          
          loss_fct = BCEWithLogitsLoss()
          batch_loss = loss_fct(logits.view(-1, self.num_labels).float(), labels.view(-1, self.num_labels).float())
          epoch_val_loss += batch_loss.item()

        avg_epoch_val_loss = epoch_val_loss/len(self.val_dataloader)
        self.val_loss_set.append(avg_epoch_val_loss)
        '''

        #Validation accuracy on the epoch
        self.mlp.eval()
        self.gmu.eval()
        epoch_f1_score = 0

        for batch_data in self.val_dataloader:
          samples_text, samples_image, labels = tuple(t.to(self.device) for t in batch_data)
          with torch.no_grad():
            attendout = self.gmu(samples_text.float(), samples_image.float())
            output = self.mlp(attendout)

          threshold = torch.Tensor([self.sigmoid_thresh]).to(self.device)
          predictions = (output > threshold).int()

          predictions = predictions.detach().cpu().numpy()
          labels = labels.to('cpu').numpy()
      
          weighted_f_score = metrics.f1_score(labels,predictions,average="weighted")
          epoch_f1_score += weighted_f_score

        avg_val_f1_score = epoch_f1_score/len(self.val_dataloader)
        print("\nWeighted F1 score for epoch: ",avg_val_f1_score,"\n")
        self.val_accuracy_set.append(avg_val_f1_score)
        
      self.Plot_Training_Epoch_Loss()
      self.Plot_Training_Epoch_SVI_Loss()
      self.Plot_Training_Epoch_Accuracy()
   

    def Test(self) :

      self.mlp.eval()
      self.gmu.eval()

      for batch_data in self.test_dataloader:
  
        samples_text, samples_image, labels = tuple(t.to(self.device) for t in batch_data)
      
        with torch.no_grad():
            attendout = self.gmu(samples_text.float(), samples_image.float())
            output = self.mlp(attendout)

        threshold = torch.Tensor([self.sigmoid_thresh]).to(self.device)
        predictions = (output > threshold).int()

        # Move preds and labels to CPU
        predictions = predictions.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()

        self.predictions = predictions
        self.Get_Metrics(labels, predictions)
        self.class_wise_metrics = metrics.classification_report(labels, predictions, target_names= list(self.label_names))
        
    
      self.results = self.results/len(self.test_dataloader)
     
      return(self.results)