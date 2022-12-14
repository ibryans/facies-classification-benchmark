import torch

class section_encoder(torch.nn.Module):
    def __init__(self, n_channels=1):
        super(section_encoder, self).__init__()
        
        self.conv_block1 = torch.nn.Sequential(
            # conv1_1
            torch.nn.Conv2d(n_channels, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # conv1_2
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # pool1
            torch.nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_1

        # 48*48

        self.conv_block2 = torch.nn.Sequential(
            # conv2_1
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # conv2_2
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # pool2
            torch.nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_2

        # 24*24

        self.conv_block3 = torch.nn.Sequential(
            # conv3_1
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # conv3_2
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # conv3_3
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # pool3
            torch.nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_3

        # 12*12

        self.conv_block4 = torch.nn.Sequential(
            # conv4_1
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # conv4_2
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # conv4_3
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # pool4
            torch.nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_4

        # 6*6

        self.conv_block5 = torch.nn.Sequential(
            # conv5_1
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # conv5_2
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # conv5_3
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # pool5
            torch.nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_5

        # 3*3

        self.conv_block6 = torch.nn.Sequential(
            # fc6
            torch.nn.Conv2d(512, 4096, 3),
            # set the filter size and nor padding to make output into 1*1
            torch.nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True), )

        # 1*1

        self.conv_block7 = torch.nn.Sequential(
            # fc7
            torch.nn.Conv2d(4096, 4096, 1),
            # set the filter size to make output into 1*1
            torch.nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True), )

    def forward(self, feature):
        sizes = dict()
        indices = dict()

        sizes[0] = feature.size()
        conv1, indices[1] = self.conv_block1(feature)
        sizes[1] = conv1.size()
        conv2, indices[2] = self.conv_block2(conv1)
        sizes[2] = conv2.size()
        conv3, indices[3] = self.conv_block3(conv2)
        sizes[3] = conv3.size()
        conv4, indices[4] = self.conv_block4(conv3)
        sizes[4] = conv4.size()
        conv5, indices[5] = self.conv_block5(conv4)
        conv6 = self.conv_block6(conv5)
        out = self.conv_block7(conv6)
        return out, indices, sizes


class section_decoder(torch.nn.Module):
    def __init__(self, n_classes=4):
        super(section_decoder, self).__init__()
        self.unpool = torch.nn.MaxUnpool2d(2, stride=2)

        self.deconv_block8 = torch.nn.Sequential(
            # fc6-deconv
            torch.nn.ConvTranspose2d(4096, 512, 3, stride=1),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True), )

        # 3*3

        self.unpool_block9 = torch.nn.Sequential(
            # unpool5
            torch.nn.MaxUnpool2d(2, stride=2), )
        # usage unpool(output, indices)

        # 6*6

        self.deconv_block10 = torch.nn.Sequential(
            # deconv5_1
            torch.nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # deconv5_2
            torch.nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # deconv5_3
            torch.nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True), )

        self.unpool_block11 = torch.nn.Sequential(
            # unpool4
            torch.nn.MaxUnpool2d(2, stride=2), )

        # 12*12

        self.deconv_block12 = torch.nn.Sequential(
            # deconv4_1
            torch.nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # deconv4_2
            torch.nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # deconv4_3
            torch.nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True), )

        self.unpool_block13 = torch.nn.Sequential(
            # unpool3
            torch.nn.MaxUnpool2d(2, stride=2), )

        # 24*24

        self.deconv_block14 = torch.nn.Sequential(
            # deconv3_1
            torch.nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # deconv3_2
            torch.nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # deconv3_3
            torch.nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True), )

        self.unpool_block15 = torch.nn.Sequential(
            # unpool2
            torch.nn.MaxUnpool2d(2, stride=2), )

        # 48*48

        self.deconv_block16 = torch.nn.Sequential(
            # deconv2_1
            torch.nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # deconv2_2
            torch.nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True), )

        self.unpool_block17 = torch.nn.Sequential(
            # unpool1
            torch.nn.MaxUnpool2d(2, stride=2), )

        # 96*96

        self.deconv_block18 = torch.nn.Sequential(
            # deconv1_1
            torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),

            # deconv1_2
            torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True), )

        self.seg_score19 = torch.nn.Sequential(
            # seg-score
            torch.nn.Conv2d(64, n_classes, 1), )

    def forward(self, feature, indices, sizes):
        conv8 = self.deconv_block8(feature) 
        conv9 = self.unpool(conv8, indices[5], output_size=sizes[4])
        conv10 = self.deconv_block10(conv9) 
        conv11 = self.unpool(conv10, indices[4], output_size=sizes[3])
        conv12 = self.deconv_block12(conv11) 
        conv13 = self.unpool(conv12, indices[3], output_size=sizes[2])
        conv14 = self.deconv_block14(conv13) 
        conv15 = self.unpool(conv14, indices[2], output_size=sizes[1])
        conv16 = self.deconv_block16(conv15) 
        conv17 = self.unpool(conv16, indices[1], output_size=sizes[0])
        conv18 = self.deconv_block18(conv17)
        out = self.seg_score19(conv18)
        return out


class section_two_stream(torch.nn.Module):
    def __init__(self, n_channels=1, n_classes=4):
        super(section_two_stream, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.spatial_encoder = section_encoder(n_channels)
        # self.texture_encoder = section_encoder(n_channels)
        self.unique_decoder = section_decoder(n_classes)
        
    def forward(self, feature):
        s_feature, s_indices, s_sizes = self.spatial_encoder(feature)
        # t_feature, t_indices, t_sizes = self.texture_encoder(feature)
        out = self.unique_decoder(s_feature, s_indices, s_sizes)
        return out
