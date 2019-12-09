class GAN():
    ''' batch1 and batch2 are pandas dataframes with batch 1 and batch 2 respectively and no metadata or index'''
    def __init__(self, batch1, batch2):
        # batches
        self.batch1 = batch1
        self.batch2 = batch2
        
        #parameters 
        self.data_size = batch1.shape[1]
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes x1 as input and generates gen_x1 (fake x2)
        x1 = Input(shape=(self.data_size,))
        gen_x1 = self.generator(x1)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated data as input and determines validity
        validity = self.discriminator(gen_x1)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(x1, validity) # gen_x1 is z
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(20, input_dim=self.data_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(15))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(20))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.data_size, activation='tanh'))
        # model.add(Reshape(self.img_shape))
        model.summary()

        x1 = Input(shape=(self.data_size,))
        x1_gen = model(x1)

        return Model(x1, x1_gen)

    def build_generator2(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.data_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(112))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.data_size, activation='tanh'))
        # model.add(Reshape(self.img_shape))
        model.summary()

        x1 = Input(shape=(self.data_size,))
        x1_gen = model(x1)

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.data_size,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        x2 = Input(shape=self.data_size)
        validity = model(x2)
        return Model(x2, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        plot_model = {"epoch":[],"d_loss":[],"g_loss":[]}
        batch1 = self.batch1.values
        batch2 = self.batch2.values

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of x1 and x2
            idx = np.random.randint(0, batch1.shape[0], batch_size)
            x1 = batch1[idx]
            x2 = batch2[idx]

            # Generate a batch of new images
            gen_x1 = self.generator.predict(x1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(x2, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_x1, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(x1, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            plot_model["epoch"].append(epoch)
            plot_model["d_loss"].append(d_loss[0])
            plot_model["g_loss"].append(g_loss)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_x2(epoch, x1)
        return plot_model

    def sample_x2(self, epoch, x1):
        r, c = 5, 5
        gen_x2 = self.generator.predict(x1)
        # can e.g. save some pca figures of the gen_x2