import flask
import pickle
#from flask_ngrok import run_with_ngrok
# Use pickle to load in the pre-trained model
model = pickle.load(open('model/auto_model.pkl','rb'))

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')
#run_with_ngrok(app)

# Set up the main route
@app.route('/', methods=['GET','POST'])
def main():   
    
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
        
        # Get the model's prediction
        
    
    if flask.request.method == 'POST':
        
        # Extract the input

        # Make for model
        result = model().write()
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                    result = result                               
                                     )
        

if __name__ == '__main__':
    app.run()