import pandas as pd
from tensorflow.keras.models import load_model
from preprocessing import preprocess_data

def predict():
    maxlen = 25
    _, _, test_pad, _, test = preprocess_data(maxlen=maxlen)

    model = load_model('model.h5')
    
    pred = model.predict(test_pad)
    
    result_df = pd.DataFrame(pred, columns=['target'])
    result_df['id'] = test['id'].values
    print(result_df.head())
    
    result_df.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == '__main__':
    predict()
