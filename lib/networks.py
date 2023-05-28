import torch
import torch.nn as nn

# pip regression, mobilenetv2
class Pip_mbnetv2(nn.Module):
    def __init__(self, mbnet, num_nb, num_lms=68, input_size=256, net_stride=32):
        super(Pip_mbnetv2, self).__init__()
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
        self.features = mbnet.features
        self.sigmoid = nn.Sigmoid()

        self.channel = 112

        self.cls_layer = nn.Conv2d(self.channel, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(self.channel, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(self.channel, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(self.channel, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(self.channel, num_nb*num_lms, kernel_size=1, stride=1, padding=0)

        nn.init.normal_(self.cls_layer.weight, std=0.001)
        if self.cls_layer.bias is not None:
            nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.nb_x_layer.weight, std=0.001)
        if self.nb_x_layer.bias is not None:
            nn.init.constant_(self.nb_x_layer.bias, 0)

        nn.init.normal_(self.nb_y_layer.weight, std=0.001)
        if self.nb_y_layer.bias is not None:
            nn.init.constant_(self.nb_y_layer.bias, 0)

    def forward(self, x):
        # x => [batchsize,3,256,256] 
        x = self.features(x) # [batchsize,96,8,8] 
        x1 = self.cls_layer(x) 
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5

class Pip_mbnetv2_precess_in(nn.Module):
    def __init__(self, mbnet, num_nb, num_lms=68, input_size=256, net_stride=32,reverse_index1=None,reverse_index2=None,max_len=None):
        super(Pip_mbnetv2_precess_in, self).__init__()
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
        self.features = mbnet.features
        self.reverse_index1 = reverse_index1
        self.reverse_index2 = reverse_index2
        self.max_len = max_len
        self.sigmoid = nn.Sigmoid()

        self.cls_layer = mbnet.cls_layer
        self.x_layer = mbnet.x_layer
        self.y_layer = mbnet.y_layer
        self.nb_x_layer = mbnet.nb_x_layer
        self.nb_y_layer = mbnet.nb_y_layer

    def visual(self,outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y):
        tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()

        lms_pred_x, lms_pred_y,lms_pred_nb_x,lms_pred_nb_y,_,_ = self.forward_new(outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y)

        lms_pred_x = lms_pred_x.view(tmp_batch,tmp_channel,-1)
        lms_pred_y = lms_pred_y.view(tmp_batch,tmp_channel,-1)
        lms_pred_nb_x = lms_pred_nb_x.view(tmp_batch,tmp_channel,-1)
        lms_pred_nb_y = lms_pred_nb_y.view(tmp_batch,tmp_channel,-1)
        tmp_nb_x = lms_pred_nb_x[:,self.reverse_index1, self.reverse_index2].view(tmp_batch,self.num_lms,self.max_len)
        tmp_nb_y = lms_pred_nb_y[:,self.reverse_index1, self.reverse_index2].view(tmp_batch,self.num_lms,self.max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=2), dim=2).view(tmp_channel,tmp_batch).unsqueeze(1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=2), dim=2).view(tmp_channel,tmp_batch).unsqueeze(1)
        lms_pred_merge = torch.cat((tmp_x,tmp_y),dim=1).transpose(0,2).transpose(1,2)

        return lms_pred_merge

    def forward_new(self, outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y):
        tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()

        outputs_cls = outputs_cls.view(tmp_batch*tmp_channel, -1)
        max_ids = torch.argmax(outputs_cls, 1)
        max_cls = torch.max(outputs_cls, 1)
        max_ids = max_ids.view(-1, 1)
        max_ids_nb = max_ids.repeat(1, self.num_nb).view(-1, 1)

        outputs_x = outputs_x.view(tmp_batch*tmp_channel, -1)
        outputs_x_select = torch.gather(outputs_x, 1, max_ids)
        outputs_x_select = outputs_x_select.squeeze(1)
        outputs_y = outputs_y.view(tmp_batch*tmp_channel, -1)
        outputs_y_select = torch.gather(outputs_y, 1, max_ids)
        outputs_y_select = outputs_y_select.squeeze(1)

        outputs_nb_x = outputs_nb_x.view(tmp_batch*self.num_nb*tmp_channel, -1)
        outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb)
        outputs_nb_x_select = outputs_nb_x_select.squeeze(1).view(-1, self.num_nb)
        outputs_nb_y = outputs_nb_y.view(tmp_batch*self.num_nb*tmp_channel, -1)
        outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb)
        outputs_nb_y_select = outputs_nb_y_select.squeeze(1).view(-1, self.num_nb)

        tmp_x = (max_ids%tmp_width).view(-1,1).float()+outputs_x_select.view(-1,1)
        tmp_y = (max_ids//tmp_width).view(-1,1).float()+outputs_y_select.view(-1,1)
        tmp_x /= 1.0 * self.input_size / self.net_stride
        tmp_y /= 1.0 * self.input_size / self.net_stride

        tmp_nb_x = (max_ids%tmp_height).view(-1,1).float()+outputs_nb_x_select
        tmp_nb_y = (max_ids//tmp_height).view(-1,1).float()+outputs_nb_y_select
        tmp_nb_x = tmp_nb_x.view(-1, self.num_nb)
        tmp_nb_y = tmp_nb_y.view(-1, self.num_nb)
        tmp_nb_x /= 1.0 * self.input_size / self.net_stride
        tmp_nb_y /= 1.0 * self.input_size / self.net_stride

        return tmp_x, tmp_y, tmp_nb_x, tmp_nb_y, outputs_cls, max_cls  
    
    def forward(self, x):
        x = self.features(x)
        x1 = self.cls_layer(x) 
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)

        results = self.visual(x1, x2, x3, x4, x5)

        return results
    
