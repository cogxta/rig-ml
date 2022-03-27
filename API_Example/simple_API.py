from flask import Flask, request
import json
from simple_Model import model
# input format = {'num1':1, 'num2':2}
# output format = 3

app = Flask(__name__)

@app.route('/status', methods = ['GET'])
def getHealthstatus():
    return "API is working"

	
@app.route('/predict', methods = ['POST'])
def predict():
    req = request.json; print(req)
    response = {'status':'', 'output':'', 'error':''}
    try:
        #need changes as per your model input
        num1 = req['num1']
        num2 = req['num2']
        output = model(num1,num2)
        
        print(output)
        response['output'] = output
        response['status'] = 200
    except Exception as ae:
           print(ae)
           err_msg = ""
           if hasattr(ae, 'message'):
               err_msg = ae.message
           else:
               err_msg = str(ae)
           response['error'] = err_msg
           response['status'] = 500
    response=json.dumps(response); print(response)
    return response   

	      
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)
