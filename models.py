import torch
import torch.nn as nn

class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class Maxout_MLP(nn.Module):
    
    def __init__(self, hidden_layer_size1, hidden_layer_size2, dropout, num_labels = 23, num_maxout_units=2):
        
        super(Maxout_MLP, self).__init__()
        self.fc1_list = ListModule(self, "fc1_")
        self.fc2_list = ListModule(self, "fc2_")
        self.hidden_layer_size1 = hidden_layer_size1
        self.hidden_layer_size2 = hidden_layer_size2
        for _ in range(num_maxout_units):
            self.fc1_list.append(nn.Linear(self.hidden_layer_size1, self.hidden_layer_size2))
            self.fc2_list.append(nn.Linear(self.hidden_layer_size2, self.hidden_layer_size2))
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = torch.nn.Sigmoid()
        self.linear = torch.nn.Linear(self.hidden_layer_size2, num_labels)  

        self.bn0 = nn.BatchNorm1d(self.hidden_layer_size1)
        self.bn1 = nn.BatchNorm1d(self.hidden_layer_size2)
        self.bn2 = nn.BatchNorm1d(self.hidden_layer_size2)

    def forward(self, x): 
        
        x = x.view(-1, self.hidden_layer_size1)
        x = self.bn0(x)
        x = self.maxout(x, self.fc1_list)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.maxout(x, self.fc2_list)
        x = self.bn2(x)
        logits = self.linear(x)  
        if(self.training) :     
            return logits
        else :
            output = self.sigmoid(logits)
            return output

    def maxout(self, x, layer_list):
        
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output


class GMU(nn.Module):

    def __init__(self, num_maxout_units = 2, hidden_layer_size = 512, text_embeddings_size = 300, img_embeddings_size = 4096, num_labels = 23, hidden_activation = None, dropout = 0.1):

        super(GMU, self).__init__()
        self.num_labels = num_labels
        self.hidden_layer_size = hidden_layer_size

        self.linear_h_text = torch.nn.Linear(text_embeddings_size, self.hidden_layer_size)
        self.linear_h_image = torch.nn.Linear(img_embeddings_size, self.hidden_layer_size)
        self.linear_z = torch.nn.Linear(text_embeddings_size + img_embeddings_size, self.hidden_layer_size)
       
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

        self.bn0 = nn.BatchNorm1d(img_embeddings_size)
        self.bn1 = nn.BatchNorm1d(text_embeddings_size)
        self.bn2 = nn.BatchNorm1d(text_embeddings_size + img_embeddings_size)

    def forward(self, text_embeddings, image_embeddings):
        
        image_embeddings = self.bn0(image_embeddings)
        image_h = self.linear_h_image(image_embeddings)
        image_h = self.tanh(image_h)

        text_embeddings = self.bn1(text_embeddings)
        text_h = self.linear_h_text(text_embeddings)
        text_h = self.tanh(text_h)

        concat = torch.cat((image_embeddings, text_embeddings), 1)
        concat = self.bn2(concat)
        z = self.linear_z(concat)
        z = self.sigmoid(z)
        gmu_output = z*image_h + (1-z)*text_h
        
        attendout = self.tanh(gmu_output)  

        return attendout