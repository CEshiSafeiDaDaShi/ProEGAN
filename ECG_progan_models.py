# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:34:39 2020


@author: intel
"""

import torch

import ECG_progan_layers
import ECG_config

class AccessibleDataParallel(torch.nn.DataParallel):
    """
    A slight modification of PyTorch's default DataParallel wrapper object; this allows
    us to access attributes of a DataParallel-wrapped module.
    """
    def __getattr__(self, name):
        try:
            # Return an attribute defined in DataParallel
            return super().__getattr__(name)
        except:
            # Otherwise return the wrapped module's attribute of the same name
            return getattr(self.module, name)

class Generator(torch.nn.Module):
    """
    An image generator as described in the ProGAN paper. This model is composed of a
    set of blocks, each of which are trained in sequence. The first block converts a 1D
    input vector into a 4x4 featuremap; all other blocks upscale by a factor of 2 and
    apply additional convolution layers. Each block uses leaky ReLU activation (0.2 * x
    for x < 0, x otherwise) and pixelwise normalization (see the Pixnorm layer).

    Each block also has a toRGB layer which converts the output of that block to
    the RGB color space.
    
    程序文件中描述的图像发生器。该模型由一组块组成，每个块按顺序训练。第一个块将1D输入矢量转换为4x4特征图；
    所有其他块将放大2倍，并应用额外的卷积层。每个块使用泄漏ReLU激活（x<0时为0.2*x，否则为x）和像素归一化（参见Pixnorm层）。
    每个块还具有一个toRGB层，该层将该块的输出转换为RGB颜色空间。
    """

    def __init__(self):
        super().__init__()
        
        self.toECGs = []
        self.blocks = []

        def new_block(block_index):
            """Returns a block; we use a trick from the ProGAN paper to upscale and
            convolve at the same time in the first layer. 
            返回一个块；我们使用ProGAN文件中的一个技巧在第一层中同时做上采样和卷积"""
            return torch.nn.Sequential(
                ECG_progan_layers.EqualizedConvTranspose1D(
                    in_channels=ECG_config.cfg.blocks[block_index - 1],
                    out_channels=ECG_config.cfg.blocks[block_index],
                    kernel_size=3,
                    padding=1,
                    output_padding =0,
                    stride=2,
                    upscale=True
                ),
               ECG_progan_layers.Pixnorm(),
                torch.nn.LeakyReLU(0.2, inplace=True),

               ECG_progan_layers.EqualizedConv1d(
                    in_channels=ECG_config.cfg.blocks[block_index],
                    out_channels=ECG_config.cfg.blocks[block_index],
                    kernel_size=3,
                    padding=1,
                ),
                ECG_progan_layers.Pixnorm(),
                torch.nn.LeakyReLU(0.2, inplace=True),
            )

        # Block 0
        self.blocks.append(
            torch.nn.Sequential(
                # This pixnorm layer converts gaussian noise inputs to "points on a
                # 512-dimensional hypersphere" as noted in the paper. Where 512 is
                # the latent space size.
                ECG_progan_layers.Pixnorm(),

                # A 4x4 transposed convolution applied to a 1x1 input will yield a 4x4 output
                # 将4x4转置卷积应用于1x1输入将产生4x4输出
                # 原文 init_res = 4，我们这里用 init_res=41，第一次输出为41的长度
                # 所以我们的第一个一维卷积核长度为41               
                # 
                ECG_progan_layers.EqualizedConvTranspose1D(
                    in_channels=ECG_config.cfg.latent_dim,
                    out_channels=ECG_config.cfg.blocks[0],
                    kernel_size=ECG_config.cfg.init_res,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),
                ECG_progan_layers.Pixnorm(),
                # 这里将128x41的数据进行卷积，输出128x41不变，但是这个kernel_size=3该选择几？
                ECG_progan_layers.EqualizedConv1d(
                    in_channels=ECG_config.cfg.blocks[0],
                    out_channels=ECG_config.cfg.blocks[0],
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),
                ECG_progan_layers.Pixnorm(),
            )
        )

        for block in range(len(ECG_config.cfg.blocks)):
            self.toECGs.append(
                ECG_progan_layers.EqualizedConv1d(
                    in_channels=ECG_config.cfg.blocks[block],
                    out_channels=1,
                    kernel_size=1,
                    padding=0,
                )
            )
            # Don't add a new block on the last iteration, because the final block was
            # already here.
            if block < len(ECG_config.cfg.blocks) - 1:
                self.blocks.append(new_block(block + 1))

        # We need to register the blocks as modules for PyTorch to register their
        # weights as model parameters to optimize.
        # 我们需要将Block注册为module，以便Pythorch将其权重注册为模型参数以进行优化。
        self.toECGs = torch.nn.ModuleList(self.toECGs)
        self.blocks = torch.nn.ModuleList(self.blocks)

    def forward(self, latent_sample, top_blocks, blend_ratio):
        features = None
        lil_toECG = self.toECGs[top_blocks[0]]
        big_block = self.blocks[top_blocks[1]] if len(top_blocks) == 2 else None
        big_toECG = self.toECGs[top_blocks[1]] if len(top_blocks) == 2 else None

        for i, block in enumerate(self.blocks):
            features = block(features) if features is not None else block(latent_sample)
            if i == top_blocks[0]:
                if len(top_blocks) == 1:
                    return lil_toECG(features)
                else:
                    trained_img = lil_toECG(features)
                    #torch.nn.functional.interpolate 进行插值上采样
                    trained_img = torch.nn.functional.interpolate(trained_img,
                                                                  scale_factor=2.0,
                                                                  mode="nearest")
                    new_img = big_toECG(big_block(features))
                    return blend_ratio * new_img + (1.0 - blend_ratio) * trained_img

    def momentum_update(self, source_model, decay):
        """
        Updates the weights in self based on the weights in source_model.
        New weights will be decay * self's current weights + (1.0 - decay)
        * source_model's weights.

        This is used to make small updates to a visualizer network, moving each weight
        slightly toward the generator's weights. Doing it this way helps reduce
        artifacts and rapid changes in generated images, since we're averaging many
        states of the generator. The visualizer is what generates the sample images
        during training, and should probably be what's used for generation after
        training is complete.

        One thing worth noting is that images will appear less stable as training
        proceeds because, as the batch size shrinks, more updates to the visualizer are
        made between samples so this feature's effect on stability is diminished
        somewhat.
        """

        # Gets a dictionary mapping each generator parameter's name to the actual
        # parameter object. If you aren't using AccessibleDataParallel because you have
        # just one GPU, remove the "module" attribute and just use
        # dict(source_model.named_parameters()) #每个对象的参数映射到一个实际的名字。
        # 如果因为只有一个GPU而没有使用AccessibleDataParallel，请删除“module”属性，只使用#dict（源代码_model.named_参数())
        param_dict_src = dict(source_model.named_parameters())


        # For each parameter in the visualization model, get the same parameter in the
        # source model and perform the update.
        with torch.no_grad():
            for p_name, p_target in self.named_parameters():
                p_source = param_dict_src[p_name].to(p_target.device)
                p_target.copy_(decay * p_target + (1.0 - decay) * p_source)


class Discriminator(torch.nn.Module):
    """
    A discriminator between generated and real input images, as described in the ProGAN
    paper. This model is composed of a set of blocks, each of which are trained in
    sequence. Block 0 is the last block data sees, outputting the discriminator's score.
    But Block 0 is also trained first.

    The final block computes the mean standard deviation of pixel values across the
    batch, and adds that as an extra feature to the input, then applies 1 convolution.
    Next, the block applies an unpadded 4x4 kernel to the resulting 4x4 featuremap,
    resulting in a 1x1 output (with 512 channels in the default configuration). A final
    convolution to a 1-channel output yields the discriminator's score of the input's
    "realness". This last convolution is equivalent to a "fully-connected" layer and is
    what the paper actually did in its code.

    As with the generator, each block contains a convolution layer plus a layer that
    applies both a convolution and downsample at the same time. The same leaky ReLU
    activation is used, but unlike the generator pixelwise normalization is not.

    Each block also has a fromRGB layer which converts an RGB image sized for that block
    into a featuremap with the number of channels expected by the block's first layer.
    
    生成图像和实际输入图像之间的一种鉴别器，如ProGAN论文中所述。
    该模型由一组块组成，每个块按顺序训练。Block0是最后看到的块数据，输出鉴别器的分数。但是Block0也是先训练的。
    最后一个块计算批处理中像素值的平均标准差，并将其作为额外特征添加到输入中，然后应用1卷积。
    接下来，块将一个未添加的4x4内核应用于生成的4x4特性映射，从而产生1x1输出（默认配置中有512个通道）。
    最后对一个单通道输出进行卷积，得到鉴别器对输入“真实性”的分数。最后一个卷积相当于一个“完全连接”的层，这也是本文在代码中所做的。
    与生成器一样，每个块包含一个卷积层和一个同时应用卷积和下采样的层。使用相同的泄漏ReLU激活，但不同于生成器像素归一化不是。
    每个块还具有fromRGB层，该层将为该块调整大小的RGB图像转换为具有块第一层所需通道数的featuremap。
    """
    def __init__(self):
        super().__init__()

        self.blocks = []
        self.fromECGs = []

        def new_block(block_index):
            return torch.nn.Sequential(
                ECG_progan_layers.EqualizedConv1d(
                    in_channels=ECG_config.cfg.blocks[block_index],
                    out_channels=ECG_config.cfg.blocks[block_index - 1],
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),

                ECG_progan_layers.EqualizedConv1d(
                    in_channels=ECG_config.cfg.blocks[block_index - 1],
                    out_channels=ECG_config.cfg.blocks[block_index - 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    downscale=True
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),
            )
        # 这是最后一个block
        self.blocks.append(
            torch.nn.Sequential(
                ECG_progan_layers.StandardDeviation(),
                
                ECG_progan_layers.EqualizedConv1d(
                    in_channels=ECG_config.cfg.blocks[0] + 1,  # +1 for std dev channel
                    out_channels=ECG_config.cfg.blocks[0],
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),

                # Input BxCx4x4; Output BxCx1x1
                ECG_progan_layers.EqualizedConv1d(
                    in_channels=ECG_config.cfg.blocks[0],
                    out_channels=ECG_config.cfg.blocks[0],
                    kernel_size=32,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),

                # Input BxCx1x1; collapsed to Bx1x1x1 - 1 score for each sample
                ECG_progan_layers.EqualizedConv1d(
                    in_channels=ECG_config.cfg.blocks[0], out_channels=1, kernel_size=1
                ),
            )
        )

        # We build the discriminator from back to front; this makes it a bit harder
        # to grok if you're examining debug info, since blocks[0] is the LAST block data
        # passes through, but it is considerably easier to read and follow the code.
        # #我们从后到前构建鉴别器；这使得在检查调试信息时更难摸索，因为block[0]是最后一个通过的块数据，读取和跟踪代码要容易得多。
        for block in range(len(ECG_config.cfg.blocks)):
            self.fromECGs.append(
                ECG_progan_layers.EqualizedConv1d(
                    in_channels=1,
                    out_channels=ECG_config.cfg.blocks[block],
                    kernel_size=1,
                    padding=0,
                )
            )
            if block < len(ECG_config.cfg.blocks) - 1:
                self.blocks.append(new_block(block + 1))

        # As with the generator, convert our lists into ModuleLists to register them.
        self.blocks = torch.nn.ModuleList(self.blocks)
        self.fromECGs = torch.nn.ModuleList(self.fromECGs)

        # We'll also need a downscale layer for blending in training.
        self.halfsize = torch.nn.AvgPool1d(2)

        # Store some constants used in normalization of real samples
#        self.mean = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5]),
#                                       requires_grad=False)
#        self.std_dev = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5]),
#                                          requires_grad=False)

    def score_validity(self, img, top_blocks, blend_ratio):
        """
        This is the meat of the forward() method. The actual forward() method includes
        a few miscellaneous steps to support data preprocessing and the loss
        computation; it calls this when it needs to get a validity score for an input.
        
        这是forward（）方法的核心。实际的forward（）方法包括一些支持数据预处理和损失计算的步骤；
        当需要获取输入的有效性分数时，它将调用此方法。
        """

        lil_fromECG = self.fromECGs[top_blocks[0]]
        big_block = self.blocks[top_blocks[1]] if len(top_blocks) == 2 else None
        big_fromECG = self.fromECGs[top_blocks[1]] if len(top_blocks) == 2 else None

        # The reverse of the generator - the layer we start training with depends on
        # which head is being trained, but we always proceed through to the end.
        if big_block is not None:
            features = big_fromECG(img)
            features = big_block(features)
            trained_features = lil_fromECG(self.halfsize(img))
            features = blend_ratio * features + (1.0 - blend_ratio) * trained_features
        else:
            features = lil_fromECG(img)
        # The list slice here steps backward from the smaller-resolution of the top
        # blocks being trained 这里的列表片段从正在训练的顶部块的较小分辨率，反向搭建
        for block in self.blocks[top_blocks[0]::-1]:
            features = block(features)

        # The view here just takes the output from Bx1x1x1 to B
        return features.view(-1)

    def forward(self, fake_img, real_imgs, top_blocks, blend_ratio):
        if real_imgs == None:
            # When we compute the generator's loss, we don't do anything fancy -
            # just return the negative of the discriminator's score 
            # 当我们计算生成器的loss时，我们不会做任何事情——只返回鉴别器分数的负数
            # score_validity这个分数到底干嘛的？
            return -torch.mean(self.score_validity(fake_img, top_blocks, blend_ratio))
        else:
            # Get the discriminator's opinion on a batch of fake images and one of real
            fake_validity = self.score_validity(fake_img, top_blocks, blend_ratio)
            real_validity = self.score_validity(real_imgs, top_blocks, blend_ratio)

            # WGAN style loss; we want the discriminator to output numbers as
            # negative as possible for fake_validity and as positive as possible
            # for real_validity
            # #WGAN风格的损失；我们希望鉴别器输出fake_validity的值尽可能负，real_validity尽可能正
            wgan_loss = torch.mean(fake_validity) - torch.mean(real_validity)

            # Add a penalty for the discriminator having gradients far from 1 on images
            # composited from real and fake images (this keeps training from wandering
            # into unstable regions).
            # 在由真实和假图像合成的图像上，如果鉴别器的梯度远离1（这会使训练不会游走到不稳定的区域中），则会增加一个惩罚。
            gradient_penalty = ECG_config.cfg.lambda_gp * self.gradient_penalty(real_imgs,
                                                                            fake_img,
                                                                            top_blocks,
                                                                            blend_ratio)

            # Add a penalty for the discriminator's score on real images drifting
            # too far from 0; this helps keep the discriminator from being too
            # confident, which can result in near-0 gradients for the generator to
            # learn from. It also keeps numbers from getting big and overflowing.
            #为鉴别器在实际图像上偏离0太远的分数添加一个惩罚；这有助于防止鉴别器过于自信，
            # 这可能导致生成器学习的梯度接近0。它还可以防止数字变得庞大和泛滥。
            drift_penalty = 0.001 * torch.mean(real_validity ** 2.0)

            return wgan_loss + gradient_penalty + drift_penalty

    def gradient_penalty(self, real_samples, fake_samples, top_blocks, blend_ratio):
        """
        Computes a penalty to the discriminator for having gradients far from 1. This is
        desirable to keep training stable since parameter updates will have sane sizes.
        For a more mathematical explanation that is, frankly, over my head, read the
        WGAN-GP paper at https://arxiv.org/abs/1704.00028

        This method interpolates each real image with a generated one, with a random
        weight. The penalty is then computed from the gradient of the input with respect
        to the discriminator's score for that interpolated image. I can't explain why.
        
        计算鉴别器对坡度远离1的惩罚。这对于保持训练的稳定性是可取的，因为参数更新的大小是合理的。
        对于一个更数学化的解释，阅读WGAN-GP的论文https://arxiv.org/abs/1704.00028
        该方法用随机加权的方法对生成的图像进行插值。然后根据输入相对于该内插图像的鉴别器分数的梯度来计算惩罚。
        我无法解释为什么。
        """

        batch_size = real_samples.size(0)

        # Random weight for interpolation between real/fake samples; one weight for each
        # sample in the batch. 用于在真/假样本之间插值的随机权重；批次中每个样本一个权重。
        # rand函数返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数
        image_weights = torch.rand((batch_size, 1, 1)).to(fake_samples.device)

        # Compute the interpolations between the real and fake samples.
        interpolated = (
            image_weights * real_samples + ((1 - image_weights) * fake_samples)
        ).requires_grad_(True)

        interpolated_validity = self.score_validity(interpolated,
                                                    top_blocks,
                                                    blend_ratio)

        # Get gradient of input with respect to the interpolated images
        gradients = torch.autograd.grad(
            outputs=interpolated_validity,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_validity),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].view(batch_size, -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty