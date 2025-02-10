from tensorflow.keras.applications import EfficientNetB4  # Better than VGG16
from tensorflow.keras.regularizers import l2

# Base model
base_model = EfficientNetB4(
    weights='imagenet',
    include_top=False,
    input_shape=(300, 300, 3)  # Increased input size
)

# Custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.005))(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Phase 1: Train the head
for layer in base_model.layers:
    layer.trainable = False

model.compile(Adam(learning_rate=1e-3), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Phase 2: Fine-tune 30% of base layers
for layer in base_model.layers[-int(0.3*len(base_model.layers)):]:
    layer.trainable = True

model.compile(Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])