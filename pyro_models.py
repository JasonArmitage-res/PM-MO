import torch
import torch.nn as nn
from torch.nn.functional import softplus
import pyro
from pyro import poutine
from pyro.distributions import Normal, Categorical, Laplace

# Pyro module defined only over the GMU NN
class GMU_Pyro():

    def __init__(self, model):

        self.model = model 
    
    def gmu_model(self, text_embeddings, image_embeddings, labels):

        pyro.module("GMU", self.model)
        #global x_label

        linear_h_text_prior_w = Laplace(loc=torch.ones_like(self.model.linear_h_text.weight), scale=torch.ones_like(self.model.linear_h_text.weight))
        linear_h_text_prior_b = Laplace(loc=torch.ones_like(self.model.linear_h_text.bias), scale=torch.ones_like(self.model.linear_h_text.bias))
        linear_h_image_prior_w = Laplace(loc=torch.ones_like(self.model.linear_h_image.weight), scale=torch.ones_like(self.model.linear_h_image.weight))
        linear_h_image_prior_b = Laplace(loc=torch.ones_like(self.model.linear_h_image.bias), scale=torch.ones_like(self.model.linear_h_image.bias))
        linear_z_prior_w = Laplace(loc=torch.ones_like(self.model.linear_z.weight), scale=torch.ones_like(self.model.linear_z.weight))
        linear_z_prior_b = Laplace(loc=torch.ones_like(self.model.linear_z.bias), scale=torch.ones_like(self.model.linear_z.bias))

        priors = {
            'linear_h_text.weight': linear_h_text_prior_w, 'linear_h_text.bias': linear_h_text_prior_b,
            'linear_h_image.weight': linear_h_image_prior_w, 'linear_h_image.bias': linear_h_image_prior_b,
            'linear_z.weight': linear_z_prior_w, 'linear_z.bias': linear_z_prior_b
            }

        lifted_module = pyro.random_module("module", self.model, priors)
        lifted_reg_model = lifted_module()

        lhat = torch.sigmoid(lifted_reg_model(text_embeddings, image_embeddings))
        pyro.sample("obs", Categorical(logits=lhat), obs=labels)

    
    def gmu_guide(self, text_embeddings, image_embeddings, labels):

        pyro.module("GMU", self.model)

        # linear_h_text
        # weight
        lwlinear_h_text = torch.empty_like(self.model.linear_h_text.weight)
        swlinear_h_text = torch.empty_like(self.model.linear_h_text.weight)
        torch.nn.init.normal_(lwlinear_h_text, std=0.001)
        torch.nn.init.normal_(swlinear_h_text, std=0.01)
        linear_h_text_loc_param_w_l = pyro.param("linear_h_text_loc_w_l", lwlinear_h_text)
        linear_h_text_loc_param_w_s = softplus(pyro.param("linear_h_text_loc_w_s", swlinear_h_text))
        linear_h_text_prior_w = Laplace(loc=linear_h_text_loc_param_w_l, scale=linear_h_text_loc_param_w_s)
        # bias
        lblinear_h_text = torch.empty_like(self.model.linear_h_text.bias)
        sblinear_h_text = torch.empty_like(self.model.linear_h_text.bias)
        torch.nn.init.normal_(lblinear_h_text, std=0.001)
        torch.nn.init.normal_(sblinear_h_text, std=0.01)
        linear_h_text_loc_param_b_l = pyro.param("linear_h_text_loc_b_l", lblinear_h_text)
        linear_h_text_loc_param_b_s = softplus(pyro.param("linear_h_text_loc_b_s", sblinear_h_text))
        linear_h_text_prior_b = Laplace(loc=linear_h_text_loc_param_b_l, scale=linear_h_text_loc_param_b_s)

        # linear_h_image
        # weight
        lwlinear_h_image = torch.empty_like(self.model.linear_h_image.weight)
        swlinear_h_image = torch.empty_like(self.model.linear_h_image.weight)
        torch.nn.init.normal_(lwlinear_h_image, std=0.001)
        torch.nn.init.normal_(swlinear_h_image, std=0.01)
        linear_h_image_loc_param_w_l = pyro.param("linear_h_image_loc_w_l", lwlinear_h_image)
        linear_h_image_loc_param_w_s = softplus(pyro.param("linear_h_image_loc_w_s", swlinear_h_image))
        linear_h_image_prior_w = Laplace(loc=linear_h_image_loc_param_w_l, scale=linear_h_image_loc_param_w_s)
        # bias
        lblinear_h_image = torch.empty_like(self.model.linear_h_image.bias)
        sblinear_h_image = torch.empty_like(self.model.linear_h_image.bias)
        torch.nn.init.normal_(lblinear_h_image, std=0.001)
        torch.nn.init.normal_(sblinear_h_image, std=0.01)
        linear_h_image_loc_param_b_l = pyro.param("linear_h_image_loc_b_l", lblinear_h_image)
        linear_h_image_loc_param_b_s = softplus(pyro.param("linear_h_image_loc_b_s", sblinear_h_image))
        linear_h_image_prior_b = Laplace(loc=linear_h_image_loc_param_b_l, scale=linear_h_image_loc_param_b_s)

        # linear_z
        # weight
        lwlinear_z = torch.empty_like(self.model.linear_z.weight)
        swlinear_z = torch.empty_like(self.model.linear_z.weight)
        torch.nn.init.normal_(lwlinear_z, std=0.001)
        torch.nn.init.normal_(swlinear_z, std=0.01)
        linear_z_loc_param_w_l = pyro.param("linear_z_loc_w_l", lwlinear_z)
        linear_z_loc_param_w_s = softplus(pyro.param("linear_z_loc_w_s", swlinear_z))
        linear_z_prior_w = Laplace(loc=linear_z_loc_param_w_l, scale=linear_z_loc_param_w_s)
        # bias
        lblinear_z = torch.empty_like(self.model.linear_z.bias)
        sblinear_z = torch.empty_like(self.model.linear_z.bias)
        torch.nn.init.normal_(lblinear_z, std=0.001)
        torch.nn.init.normal_(sblinear_z, std=0.01)
        linear_z_loc_param_b_l = pyro.param("linear_z_loc_b_l", lblinear_z)
        linear_z_loc_param_b_s = softplus(pyro.param("linear_z_loc_b_s", sblinear_z))
        linear_z_prior_b = Laplace(loc=linear_z_loc_param_b_l, scale=linear_z_loc_param_b_s)

        priors = {
        'linear_h_text.weight': linear_h_text_prior_w, 'linear_h_text.bias': linear_h_text_prior_b,
        'linear_h_image.weight': linear_h_image_prior_w, 'linear_h_image.bias': linear_h_image_prior_b,
        'linear_z.weight': linear_z_prior_w, 'linear_z.bias': linear_z_prior_b
        }

        lifted_module = pyro.random_module("module", self.model, priors)

        return lifted_module()