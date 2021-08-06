# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:46:55 2020

@author: intel
"""

import torch
import numpy


class Pixnorm(torch.nn.Module):
    """
     为了提高训练的稳定性，加入了 pixel-wise normalization
    Normalizes the input tensor to magnitude 1 without changing direction;
    as a different perspective, maps inputs to points on an N-dimensional sphere
    with radius 1, where N is the number of input dimensions.
    第二句话有疑问
    """

    def forward(self, in_features):
        channels = in_features.size()[1]  # Inputs are batch x channel x lenth
        normalizer = torch.sqrt(
            1e-8 + torch.sum(in_features ** 2.0, dim=1, keepdim=True) / channels
        )
        return in_features.div(normalizer)


class StandardDeviation(torch.nn.Module):
    """
    不使用 batchnorm，为了提高生成器模型生成数据的多样性，在判别器的最后一层上加入了 minibatch discrimination
    Adds an extra channel to the input, containing the standard deviation of
    pixel values across the batch. The new channel contains the mean of each
    pixel's standard deviation. This gives the discriminator a clue to detect
    generated batches if the generator only learns how to generate one image,
    which forces the generator to learn a distribution of images to generate.
    给输出增加一个额外的批次标准差的通道输入， 
    这给鉴别器提供了一条线索，如果生成器只学习如何生成一个图像，
    那么这个标准差的输入会让判别其检器检测生成的批次，这迫使生成器学习生成图像的分布。
    """

    def forward(self, in_features):
        batch_size, _, length  = in_features.shape

        # B x 1 x L; 减去批次平均值。
        output = in_features - in_features.mean(dim=0, keepdim=True)

        # 1 x L; 计算样本中标准差
        output = torch.sqrt_(output.pow_(2.0).mean(dim=0, keepdim=False) + 10e-8)

        # 1 x 1 x 1 ; 取特征图和像素的平均值。
        output = output.mean().view(1, 1, 1)

        # B x 1 x L; 在组和像素上复制。
        output = output.repeat(batch_size, 1, length)

        # Append that channel to the original input
        output = torch.cat([in_features, output], 1)
        return output


class EqualizedConv1d(torch.nn.Module):
    """
    EqualizedConv2d 均衡以1维卷积，普通卷积对权重增加了缩放。
    This is mainly a standard convolutional layer, but with the added feature
    that its weights are scaled by the constant given in He 
    (https://arxiv.org/abs/1502.01852). The variance in gradients for a layer
    depends on the size of its input; scaling to account for this helps smooth
    out training and helps ensure the model converges to something. 
    
    这个论文细看一下，重点在MSRA初始化这部分。
    这主要是一个标准的卷积层，但是它对权重进行了缩放。利用何凯明论文中给定的MSRA方法来缩放(https://arxiv.org/abs/1502.01852).
    一个层的梯度变化取决于其输入的大小；缩放有助于平滑训练，助于确保模型收敛。
    
    The linked paper only applies scaling at initialization; this layer instead
    initializes without scaling, and does the scaling at every forward pass. This still
    helps account for the variance during training, but since the weight's value isn't
    changed by scaling, a large scaling factor won't reduce the value to nearly 0 and
    thus won't ensure the gradients with respect to that weight are nearly 0 no matter
    what the loss happens to be.
    
    链接的论文中仅在初始化时应用缩放；
    这里我们在层在没有缩放的情况下初始化，并在每次向前传递时进行缩放。
    这仍然有助于解释训练过程中的方差，但由于权重值不会因缩放而改变，因此较大的缩放因子不会将该值降低到接近0，
    因此，无论损失是什么，都不能保证相对于该权重的梯度接近0。
    （不太理解 需要细看论文中的解释）
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0,
        downscale=False
    ):
        super().__init__()

        # Create an empty tensor for the filter's weights, then initialize it

        self.weights = torch.nn.Parameter(
            torch.nn.init.normal_(  # Underscore suffix for in-place operation
                torch.empty(out_channels, in_channels, kernel_size)
            ),
            requires_grad=True,
        )

        self.stride = stride
        self.padding = padding
        self.downscale = downscale

        self.bias = torch.nn.Parameter(torch.FloatTensor(out_channels).fill_(0))

        # He initialization scaling factor
        fan_in = kernel_size * in_channels
        self.scale = numpy.sqrt(2) / numpy.sqrt(fan_in)

    def forward(self, in_features):
        if self.downscale:
            # Pad the last  dimensions (L) of the kernel weights
            #填充核权重的最后一个维度（L）
            weight = torch.nn.functional.pad(self.weights, [1, 1])
            # Blur the weights by averaging the 4 4x4 corners; if we didn't do this
            # every second weight-pixel pair would be skipped with the stride of 2.
            #＃通过平均2个前段与后段角来模糊权重； 如果我们不这样做，则第二个权重像素对将被跳过2步。
            weight = (  weight[:, :,  :-1]
                      + weight[:, :, 1: ]) / 2.0
        else:
            weight = self.weights
        return torch.nn.functional.conv1d(
            input=in_features,
            weight=weight * self.scale,  # Scaling weights dynamically
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class EqualizedConvTranspose1D(torch.nn.Module):
    """
    This is a transpose convolution layer, modified to support dynamic scaling
    in the same way as the above layer. For an in-depth guide to the behavior of
    transposed convolutions, https://arxiv.org/pdf/1603.07285v1.pdf is a good source
    for some intuition, as is
    https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8
    but for us the key takeaway is that transpose convolution behaves like a
    typical convolution, but padded with zeros in specific ways.

    Adding strides to a transpose convolution spreads where the kernel is applied to the
    output, rather than the input - that is, you still apply the kernel to each input
    pixel, but when you move from one input pixel to the next, the kernel's output
    in the output image will skip several pixels instead of 1. This upscales the output.

    Additionally, "padding" a transpose convolution means discarding data from the edges
    of the output, rather than adding empty data to the edges of the input. With no
    padding, you get a larger output even without striding (that's how
    an unpadded 1x1 input from our latent space is converted to a starting 4x4), while
    the traditional amount of padding to keep the output the same size in a regular
    convolution does the same thing in a transpose convolution.

    There are gotchas with transpose convolutions that can lead to bad artifacting in
    generated images, so if you want to tweak the kernel sizes, I recommend delving into
    the math a bit. https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, upscale=False):
        super().__init__()

        self.weights = torch.nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(in_channels, out_channels, kernel_size)
            )
        )

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.bias = torch.nn.Parameter(torch.FloatTensor(out_channels).fill_(0))

        # Fan-in is simply number of input channels
        self.scale = numpy.sqrt(2) / numpy.sqrt(in_channels)

        self.upscale = upscale

    def forward(self, in_features):
        if self.upscale:
            # Pad the last two dimensions (L) of the kernel weights
            weight = torch.nn.functional.pad(self.weights, [1, 1])

            # Blur the weights by summing the 4 4x4 corners; I think we sum rather than
            # average because of the 0s inserted into the source tensor by strided
            # transposed convolution, but to be honest haven't sat down to do the
            # algebra yet.
            weight = (  weight[:, :,  :-1]
                      + weight[:, :, 1: ]) 
        else:
            weight = self.weights
        return torch.nn.functional.conv_transpose1d(
            input=in_features,
            weight=weight * self.scale,  # Scaling weights dynamically,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
