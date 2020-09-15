from tensorflow.keras import Model, Input, Dense


class OptionNetwork:
    def __init__(self, name=''):
        inputs = Input(shape=(10,))
        a1 = Dense(20, activation='relu')(inputs)
        a2 = Dense(20, activation='relu')(a1)
        low_level_action = Dense(4, activation='softmax')(a2)
        is_terminate_output = Dense(1, activation='sigmoid')(a2)

        self.model = Model(
            inputs=inputs,
            outputs=[
                low_level_action,
                is_terminate_output
            ],
            name=name
        )

    def freeze(self):
        self.model.trainable = False
    
    def unfreeze(self):
        self.model.trainable = True

    def receive_reward(self, reward, tape, optimizer):
        pass
