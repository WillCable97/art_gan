from Runners.Runner import Runner
import tensorflow as tf

"""GAN instance of runner class, implements its own run_step method"""

class CycleGanRunner(Runner):
    def run_step(self, inputs):
        LAMBDA=self.model_hyperparams.LAMBDA.value
        
        doma_eg = inputs[0]
        domb_eg = inputs[1]
        
        with tf.GradientTape(persistent=True) as tape:
            #Generate domain transfer
            fake_doma_eg = self.model_collection['b_a_generator'](domb_eg)
            fake_domb_eg = self.model_collection['a_b_generator'](doma_eg)
            
            #Measure of how generated image performs against decriminator
            fake_doma_eg_sucess = self.model_collection['a_descrim'](fake_doma_eg)
            fake_domb_eg_sucess = self.model_collection['b_descrim'](fake_domb_eg)
            real_doma_eg_sucess = self.model_collection['a_descrim'](doma_eg)
            real_domb_eg_sucess = self.model_collection['b_descrim'](domb_eg)

            #Handle descriminator losses
            doma_descrim_loss = self.descriminator_loss(real_doma_eg_sucess, fake_doma_eg_sucess)
            domb_descrim_loss = self.descriminator_loss(real_domb_eg_sucess, fake_domb_eg_sucess)
            
            #Handfle generator losses
            fake_doma_descrim_loss = self.entropy_loss(tf.ones_like(fake_doma_eg_sucess), fake_doma_eg_sucess)
            fake_domb_descrim_loss = self.entropy_loss(tf.ones_like(fake_domb_eg_sucess), fake_domb_eg_sucess)
            
            #Feed fake images back to origina domain
            doma_cycled = self.model_collection['b_a_generator'](fake_domb_eg)
            domb_cycled = self.model_collection['a_b_generator'](fake_doma_eg)
            total_cyclic_loss = tf.reduce_mean(tf.abs(doma_eg - doma_cycled)) + tf.reduce_mean(tf.abs(domb_eg - domb_cycled))

            #Identity loss (mapping to same domain, would expect minimal or no change)
            doma_identity = self.model_collection['b_a_generator'](doma_eg)
            domb_identity = self.model_collection['a_b_generator'](domb_eg)
            
            doma_identity_loss = tf.reduce_mean(tf.abs(doma_eg - doma_identity))
            domb_identity_loss = tf.reduce_mean(tf.abs(domb_eg - domb_identity))

            a_b_generator_loss = self.generator_loss(fake_domb_eg_sucess) + total_cyclic_loss * LAMBDA + domb_identity_loss * 0.5*LAMBDA
            b_a_generator_loss = self.generator_loss(fake_doma_eg_sucess) + total_cyclic_loss * LAMBDA + doma_identity_loss * 0.5*LAMBDA

        doma_descrim_gradients = tape.gradient(doma_descrim_loss, self.model_collection['a_descrim'].trainable_variables)
        self.input_optimizers['a_descrim'].apply_gradients(zip(doma_descrim_gradients, self.model_collection['a_descrim'].trainable_variables))

        domb_descrim_gradients = tape.gradient(domb_descrim_loss, self.model_collection['b_descrim'].trainable_variables)
        self.input_optimizers['b_descrim'].apply_gradients(zip(domb_descrim_gradients, self.model_collection['b_descrim'].trainable_variables))

        a_b_generator_gradients = tape.gradient(a_b_generator_loss, self.model_collection['a_b_generator'].trainable_variables)
        self.input_optimizers['a_b_generator'].apply_gradients(zip(a_b_generator_gradients, self.model_collection['a_b_generator'].trainable_variables))

        b_a_generator_gradients = tape.gradient(b_a_generator_loss, self.model_collection['b_a_generator'].trainable_variables)
        self.input_optimizers['b_a_generator'].apply_gradients(zip(b_a_generator_gradients, self.model_collection['b_a_generator'].trainable_variables))

        return [a_b_generator_loss.numpy(), b_a_generator_loss.numpy(), doma_descrim_loss.numpy(), domb_descrim_loss.numpy()]


    def descriminator_loss(self, real_example, fake_example):
        real_loss = self.entropy_loss(tf.ones_like(real_example), real_example)
        fake_loss = self.entropy_loss(tf.zeros_like(fake_example), fake_example)
        return (real_loss + fake_loss) * 0.5
    
    def generator_loss(self, descrim_output):
        return self.entropy_loss(tf.ones_like(descrim_output), descrim_output)

    def image_distance(self, image1, image2):
        pass #To do later