# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 09:55:45 2020
 
N中原始数据信息 (74546,)
N中数据筛选后信息 (74517, 256)

L中原始数据信息 (8075,)
L中数据筛选后信息 (8072, 256)

R中原始数据信息 (7259,)
R中数据筛选后信息 (7255, 256)

V中原始数据信息 (6903,)
V中数据筛选后信息 (6902, 256)

A中原始数据信息 (2546, 256)
A中数据筛选后信息 (2546, 256)
 
@author: intel
"""

import torch
import torchvision
import numpy
import copy, os, datetime
import matplotlib.pyplot as plt
from torchsummary import summary

import ECG_config
import ECG_progan_models
import ECG_progan_dataloader

N = 'N.npy'
L = 'L.npy'
R = 'R.npy'
V = 'V.npy'
A = 'A.npy'
category = N
G_loss = []
D_loss = []
ECG_path_Hz ="F:/Data/MIT-BIH-Arrhythmia/ECG_256Hz/" 
ECG_mean,ECG_std = ECG_progan_dataloader.ECG_Std(ECG_path_Hz,category)

#确保可视化样本的目录存在
os.makedirs(ECG_config.cfg.sample_location, exist_ok=True)

####################
# 模型初始化 #
####################

generator = ECG_progan_models.Generator()
discriminator = ECG_progan_models.Discriminator()

"""
制作生成器的副本，我们将使用它生成示例输出，以可视化培训进度。
在每批处理之后，可视化工具的权重将沿当前生成器权重的方向稍微更新。
我们将禁用在可视化工具中跟踪渐变，因为我们从不训练可视化工具本身。
我们将在第二个GPU上保留可视化工具，因为Pytorch在第一个GPU上使用了额外的空间。
因为我们从不计算梯度，所以它占用的空间很小，很容易拟合。
如果您只有一个GPU，考虑将其移动到CPU以提高性能，但在GPU上有更多的空间。
"""
# 需要修改设备放置在cpu上
visualizer = copy.deepcopy(generator).to(device="cuda")
for p in visualizer.parameters():
    p.requires_grad_(False)


# 尝试加载以前的训练结果，如果失败，则重新开始 #
try:
    pretrained = torch.load(ECG_config.cfg.load_location)
except:
    pretrained = None

if pretrained is None:
    # 重新开始训练
    start_at = 0

#给可视化训练模型生成一个单一的样本输入
#这样我们就可以观察网络在恒定输入下的行为。值介于0和1之间，张量为BxCx1x1，
#其中B是我们生成的样本图像的数量，C是潜在空间的大小。
    visualizer_sample = torch.FloatTensor(
        numpy.random.normal(
            0, 1, (ECG_config.cfg.sample_layout[0],
                   ECG_config.cfg.latent_dim, 1,),
        )
    ).to(device='cuda')
            
    print('visualizer_sample size:',visualizer_sample.size())
else:
    # 载入预训练权重
    generator.load_state_dict(pretrained["generator"])
    discriminator.load_state_dict(pretrained["discriminator"])
    visualizer.load_state_dict(pretrained["visualizer"])

    # Load the visualizer sample we used earlier for continuity in visualized samples
    visualizer_sample = pretrained["visualizer_sample"]

    # Note the resolution step we start training at
    start_at = pretrained["start_at"]

#####################################################
          #完成模型并行化，建立优化程序 #
#####################################################
          #这个类别信息用于确定地址的最后一段文件名，数据载入器和求均值时候确定类型地址用

# Constants for normalization, used for visualization
#img_mean = torch.tensor(ECG_mean, requires_grad=False).to(visualizer_sample.device)
#img_std_dev = torch.tensor(ECG_std, requires_grad=False).to(visualizer_sample.device)

#应用AccessibleDataParallel包装器跨GPU并行化模型；如果只有一个GPU，则可以删除这些行。
#generator = progan_models.AccessibleDataParallel(generator, (0, 1)).cuda()
#discriminator = progan_models.AccessibleDataParallel(discriminator, (0, 1)).cuda()
#将网络放置在CPU上
generator = generator.cuda()
discriminator = discriminator.cuda()


#设置优化器
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=ECG_config.cfg.lr, betas=(ECG_config.cfg.b1, ECG_config.cfg.b2)
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=ECG_config.cfg.lr, betas=(ECG_config.cfg.b1, ECG_config.cfg.b2)
)

############
 # 训练 ! #
############

images_seen = 0
runtimes = [] if pretrained is None else pretrained["runtimes"]

print('start train!')
for res_step in range(len(ECG_config.cfg.blocks)):
    # 这个 res_step 是循环表示第几个block，我们的block长度为4，所以res_step为 0 1 2 3
    # 跳过已经训练过的模型的分辨率
    if res_step < start_at:
        continue

    start = datetime.datetime.now()
    batch_size = ECG_config.cfg.batch_sizes[res_step]

    # At each resolution, we first fade that resolution's block in on top of the last
    # block, then stabilize with the newly faded-in block. top_blocks holds the two
    # blocks we need to keep track of that; the top two blocks during a fade-in and just
    # the top block when we are stabilizing. "top" here meaning the largest-resolution
    # so far involved in training.
    # Eg.res_step=5,  top_blocks输出为：[4, 5]
    #                                  [5]
    print('start block!')
    for top_blocks in ([res_step - 1, res_step], [res_step]):
        # 在这个训练步骤时，图像的实际分辨率。
        # resolution表示ECG信号的长度，res_step = 0时候，resolution等于32，之后逐步扩大2倍
        # resolution 在dataload中用来确定图像路径地址，resolution = 32 64 128 256
        resolution = ECG_config.cfg.init_res * (2 ** res_step)

        # Skip the blending phase of training the first block, since it has nothing to be blended into.
        # 如果现在是第一个block，就跳过训练的混合阶段，因为它没有什么可以混合的
        if res_step == 0 and len(top_blocks) == 2:
            continue

        # 配置dataloader以提供适当分辨率的图像。
        # 这个地方source_directory是256Hz的图像地址
        # resize_directory是降采样后的图像地址，但是降采样后的地址有128 64 32 三个地址，
        # 所以要看在训练中怎么加进去，应该是在读取之前加一个与resolution相关的地址索引函数
        # 这个地方的img_mean和img_std_dev是tensor,注意一下会不会报错
        dataloader = torch.utils.data.DataLoader(
            ECG_progan_dataloader.ECG_Dataset(
                source_directory=ECG_config.cfg.data_location,
                resolution=resolution,
                category=category
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,   # Thread count; should be about the number of CPU cores"线程计数；应该是CPU核心的数量"
            pin_memory=True,  # Faster data transfer to GPU "更快地将数据传输到GPU"
            # Set drop_last to True since a batch size mismatch between real/fake images
            # breaks the gradient penalty computation without extra code to check.
            ##将drop_last设置为True，因为真/假图像之间的批大小不匹配会中断梯度惩罚计算，而无需额外检查代码。
            # drop_last告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留
            drop_last=True,
        )

        ############
        # Training #
        ############
        for epoch in range(ECG_config.cfg.n_epochs):
            print('epoch:',epoch)
            for i, imgs in enumerate(dataloader):
                #设置高分辨率/低分辨率图像之间的混合比率。随着时间的推移，这会在高分辨率块中慢慢消失。
                # blend_ratio 混合比"
                blend_ratio = (i + epoch * len(dataloader)) / (
                    ECG_config.cfg.n_epochs * len(dataloader)
                )
                imgs=imgs[:,numpy.newaxis,:]
                imgs = imgs.type(torch.FloatTensor)  # 转Float
                real_imgs = imgs.cuda() #放在GPU设备上
                # 有疑问
                # Simulate the gradual de-blockification of input images as the new
                # resolution gets phased in, so the generator's not tasked with
                # generating higher-resolution images than it's actually capable of
                # during early blending. Failure to do this could lead to extreme
                # outputs in the new block early in blending to compensate for the new
                # block's small contribution to the output image
                #模拟输入图像随着新分辨率的逐步降低而逐渐去块化，因此，生成器的任务不是生成比早期混合时更高分辨率的图像。
                #如果不这样做，可能会导致新块在混合早期出现极端输出，以补偿新块对输出图像的小贡献
                if len(top_blocks) == 2:
                    small_imgs = torch.nn.functional.avg_pool1d(real_imgs, 2)
                    # 插值上采样
                    small_imgs = torch.nn.functional.interpolate(
                        small_imgs, scale_factor=2.0, mode="nearest"
                    )
                    real_imgs = (
                        blend_ratio * real_imgs + (1.0 - blend_ratio) * small_imgs
                    )

                #################
                # Discriminator #
                #################

                # Zero out gradients from the previous batch; the update from those
                # gradients has already been applied and the gradients are no longer valid.
                optimizer_D.zero_grad()

                #输入生成器的采样噪声。此示例是所谓的潜在空间（默认为512维空间）中的一个点。
                #生成器的工作是将空间中的点映射到图像上；我们希望维度与图像特征相对应，但我们当然不能保证这一点。
                latent_sample = torch.cuda.FloatTensor(
                    numpy.random.normal(0, 1, (batch_size, ECG_config.cfg.latent_dim,1))
                )

                # The discriminator's training doesn't depend on the gradients of the
                # generator - so we can save some space for free by not storing them.
                with torch.no_grad():
                    # 这里生成的fake_imgs尺寸为（batch,1,nHz）例如(8,1,32)
                    fake_imgs = generator(latent_sample, top_blocks, blend_ratio)

                d_loss = discriminator(fake_imgs, real_imgs, top_blocks, blend_ratio)

                #鉴别器计算了每个GPU上样本的平均值，但是我们仍然需要合并GPU结果，因此下面的mean（）就是这样。
#                d_loss = torch.mean(d_loss)
                d_loss.backward()
                optimizer_D.step()

                # Get just the numerical loss once we no longer need the graph; frees
                # up the memory before we compute the generator update.
                #（把损失留下，把graph释放）,.item()表示获取tensor中的元素值
                d_loss = d_loss.item()
                #############
                # Generator #
                #############

                # This looks the same as the process as above; there are secretly extra
                # steps in computing the loss for the discriminator, but they are built
                # into the discriminator model to parallelize them across GPUs.
                # 这看起来与上面的过程相同；在计算鉴别器的损耗时有一些秘密的额外步骤，但它们被构建到鉴别器模型中，以便在GPU上并行化。
                optimizer_G.zero_grad()
                # 这里生成的fake_imgs尺寸为Bx1xL
                fake_imgs = generator(latent_sample, top_blocks, blend_ratio)
                g_loss = discriminator(fake_imgs, None, top_blocks, blend_ratio)
#                g_loss = torch.mean(g_loss)
                g_loss.backward()
                optimizer_G.step()
                g_loss = g_loss.item()


                #无梯度状态下，对可视化部分进行操作，（sample_interval，采样间隔）
                with torch.no_grad():

                    # Update the visualizer weights slightly toward the generator
                    visualizer.momentum_update(generator, ECG_config.cfg.visualizer_decay)

                    # 有疑问
                    # Print out a variety of diagnostic information and generate some
                    # sample images every config.cfg.sample_interval image samples.
                    # Since the sample interval doesn't have to be divisible by the
                    # batch size, we're technically checking if at _least_
                    # config.cfg.sample_interval images have been shown to the model.
                    # 每训练sample_interval个样本之后，进行信息输出，并进行可视化
                    old_modulo = images_seen % ECG_config.cfg.sample_interval  
                    images_seen += batch_size 
                    new_modulo = images_seen % ECG_config.cfg.sample_interval
                            # 每个epoch记录一次损失变化
            G_loss.append(g_loss)
            D_loss.append(d_loss)
        # 每个epoch之后打印一次信息
        print(
            "[Time: {!s}][Resolution {:04d}, "
            "Top Blocks {!s}][Epoch {:03d}/{:03d}]"
            "[Batch {:05d}/{:05d}][D loss: {:.4f}][G loss: {:.4f}]"
            "[Blend: {:.4f}]".format(
                str(datetime.datetime.now() - start),
                resolution,
                top_blocks,
                epoch+1,
                ECG_config.cfg.n_epochs,
                i + 1,
                len(dataloader),
                d_loss,
                g_loss,
                blend_ratio if len(top_blocks) == 2 else 1.0,
            )
        )
        # 生成图像
        sample_imgs = visualizer(visualizer_sample,
                                 top_blocks,
                                 blend_ratio)
        print('sample_imgs size:',sample_imgs.size())
        # 记录loss
        G_loss.append(g_loss)
        D_loss.append(d_loss)
        # Manually denormalize to get a visualization that doesn't
        # depend on the range of pixel values in the generated images,
        # which it would if we used the automatic normalization in
        # the torchvision save_image function.
        #反规范化，获得不依赖于生成图像像素值范围的可视化图像
#                        sample_imgs.mul_(img_std_dev[:, None, None])
#                        sample_imgs.add_(img_mean[:, None, None])

        # Upscale the samples to the highest resolution the model will ever produce 
        # to make comparison easier.
        # 将样本放大到模型所能产生的最高分辨率，以便于比较。
        #计算长度
        final_res = ECG_config.cfg.init_res * (
            2 ** (len(ECG_config.cfg.blocks) - 1)
        )
        
        #进行上采样,scaled_samples的尺寸为torch.Size([4, 1, 256])
        scaled_samples = torch.nn.functional.interpolate(
            sample_imgs.data,
            size=(final_res),
            mode="nearest"
        )
        scaled_samples = scaled_samples.cpu() 
        scaled_samples = numpy.array(scaled_samples)
        scaled_samples = scaled_samples[0][0]
        
        #绘制生成ECG
        plt.plot(range(len(scaled_samples)), scaled_samples)
        # 生成名字
        filename = "/ECG-{:04d}-{:s}-{:03d}-{:05d}.png".format(
            resolution,
            "0" if len(top_blocks) == 2 else "1",
            epoch+1,
            i+1
        )

        # 保存图像
        plt.savefig(ECG_config.cfg.sample_location + filename, dpi=300)
        plt.close()


    # After training at each resolution, store the amount of time spent at that
    # resolution and save the models.
    # 在每个分辨率下进行训练后，存储在该分辨率下花费的时间量并保存模型。
    runtimes.append(datetime.datetime.now() - start)

#    to_save = {"generator": generator.state_dict(),
#               "discriminator": discriminator.state_dict(),
#               "visualizer": visualizer.state_dict(),
#               "visualizer_sample": visualizer_sample,
#               "start_at": res_step + 1,
#               "runtimes": runtimes,
#               }
#    
#    torch.save(to_save, ECG_config.cfg.save_location)

# As a last touch, print the runtime spent at each resolution.
# 最后一步，打印在每个分辨率下花费的运行时。
for resolution in range(len(ECG_config.cfg.blocks)):
    print(
        "Runtime for resolution %d: " % (ECG_config.cfg.init_res * (2 ** resolution)),
        runtimes[resolution],
    )


plt.plot(range(len(G_loss)), G_loss)
plt.plot(range(len(D_loss)), D_loss)


