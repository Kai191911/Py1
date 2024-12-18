import pandas as pd
import statsmodels.api as sm
import numpy as np
from flask import Flask, request, jsonify, render_template

# แก้ไฟล์
file_path = 'D:\money_eieiei\gold_price.xlsx'
df = pd.read_excel(file_path)

data = {
    'Gold_Price': df['ราคาทอง (บาท)'].tolist(),
    'Dollar_Price_Buy': df['ราคาซื้อดอลลาร์สหรัฐ'].tolist(),
    'Pound_Price_Buy': df['ราคาซื้อปอนด์'].tolist(),
    'Yen_Price_Buy': df['ราคาซื้อเยน'].tolist(),
    'Yuan_Price_Buy': df['ราคาซื้อหยวน'].tolist(),
    'Dollar_Price_Sell': df['ราคาขายดอลลาร์สหรัฐ'].tolist(),
    'Pound_Price_Sell': df['ราคาขายปอนด์'].tolist(),
    'Yen_Price_Sell': df['ราคาขายเยน'].tolist(),
    'Yuan_Price_Sell': df['ราคาขายหยวน'].tolist()
}

df = pd.DataFrame(data)

X = df[['Gold_Price']]
X = sm.add_constant(X)

Y_dollar = df['Dollar_Price_Buy']
Y_pound = df['Pound_Price_Buy']
Y_yen = df['Yen_Price_Buy']
Y_yuan = df['Yuan_Price_Buy']  # สกุลเงิน CNY
X_dollar = df['Dollar_Price_Sell']
X_pound = df['Pound_Price_Sell']
X_yen = df['Yen_Price_Sell']
X_yuan = df['Yuan_Price_Sell']

# สร้างโมเดล OLS สำหรับสกุลเงิน USD
model_dollar = sm.OLS(Y_dollar, X).fit()
B_0_dollar = model_dollar.params[0]
B_1_dollar = model_dollar.params[1]
E_dollar = np.mean(model_dollar.resid)

model_dollar = sm.OLS(X_dollar, X).fit()
B_2_dollar = model_dollar.params[0]
B_3_dollar = model_dollar.params[1]
F_dollar = np.mean(model_dollar.resid)

# สร้างโมเดล OLS สำหรับสกุลเงิน GBP
model_pound = sm.OLS(Y_pound, X).fit()
B_0_pound = model_pound.params[0]
B_1_pound = model_pound.params[1]
E_pound = np.mean(model_pound.resid)

model_pound = sm.OLS(X_pound, X).fit()
B_2_pound = model_pound.params[0]
B_3_pound = model_pound.params[1]
F_pound = np.mean(model_pound.resid)

# สร้างโมเดล OLS สำหรับสกุลเงิน JPY
model_yen = sm.OLS(Y_yen, X).fit()
B_0_yen = model_yen.params[0]
B_1_yen = model_yen.params[1]
E_yen = np.mean(model_yen.resid)

model_yen = sm.OLS(X_yen, X).fit()
B_2_yen = model_yen.params[0]
B_3_yen = model_yen.params[1]
F_yen = np.mean(model_yen.resid)

# สร้างโมเดล OLS สำหรับสกุลเงิน CNY
model_yuan = sm.OLS(Y_yuan, X).fit()
B_0_yuan = model_yuan.params[0]
B_1_yuan = model_yuan.params[1]
E_yuan = np.mean(model_yuan.resid)

model_yuan = sm.OLS(X_yuan, X).fit()
B_2_yuan = model_yuan.params[0]
B_3_yuan = model_yuan.params[1]
F_yuan = np.mean(model_yuan.resid)

# Initialize Flask app
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    currencies = ['ดอลลาร์สหรัฐ', 'ปอนด์สเตอร์ลิง', 'เยน', 'หยวน']  # เพิ่มสกุลเงินตามที่ต้องการ
    return render_template('index.html', currencies=currencies)

# Define route for forecast result
@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Get user input from form
        X_1 = float(request.form['gold_price'])
        currency = request.form['currency']  # รับค่าสกุลเงินที่เลือกจากฟอร์ม

        # Perform forecast calculation based on selected currency
        if currency == 'ดอลลาร์สหรัฐ':
            forecast_result = B_0_dollar + B_1_dollar * X_1 + E_dollar
            forecast_sell = B_2_dollar + B_3_dollar * X_1 + F_dollar
        elif currency == 'ปอนด์สเตอร์ลิง':
            forecast_result = B_0_pound + B_1_pound * X_1 + E_pound
            forecast_sell = B_2_pound + B_3_pound * X_1 + F_pound
        elif currency == 'เยน':
            forecast_result = B_0_yen + B_1_yen * X_1 + E_yen
            forecast_sell = B_2_yen + B_3_yen * X_1 + F_yen
        elif currency == 'หยวน':
            forecast_result = B_0_yuan + B_1_yuan * X_1 + E_yuan
            forecast_sell = B_2_yuan + B_3_yuan * X_1 + F_yuan
        else:
            return render_template('error.html', error_message='Invalid currency selected')

        # Return result to the same template
        return render_template('index.html', currencies=['ดอลลาร์สหรัฐ', 'ปอนด์สเตอร์ลิง', 'เยน', 'หยวน'], forecast_result=forecast_result,forecast_sell=forecast_sell, currency=currency)

    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
