import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('data.csv')

# Select the features and target variables
features = data[['max', 'wind', 'pressure', 'humidi']].values
target = data[['rain']].values

# Scale the features to a range between 0 and 1
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2)

# Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy:', accuracy)

# Make predictions
new_data = [[30, 8, 1012, 50]]  # Example input: temperature=30, wind=8, pressure=1012, humidity=50
scaled_new_data = scaler.transform(new_data)
prediction = model.predict(scaled_new_data)
print('Predicted Rain:', prediction)