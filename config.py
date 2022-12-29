self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
self.cnet = BasicEncoder(output_dim=256, norm_fn='batch', dropout=args.dropout)

fmap1, fmap2 = self.fnet([image1, image2])

train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                               pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
