# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

# Python modules
import os, logging 
import random
# Flask modules
from flask               import render_template, request, url_for, redirect, send_from_directory,jsonify,session,send_file
from flask_login         import login_user, logout_user, current_user, login_required
from werkzeug.exceptions import HTTPException, NotFound, abort
from jinja2              import TemplateNotFound

# App modules
from app        import app, lm, db, bc
from app.models import *

#from app.forms  import LoginForm, RegisterForm
import requests
import os
import openai
import cv2
import numpy as np
from dotenv import load_dotenv
import base64
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
API_URL_trocr="https://api-inference.huggingface.co/models/microsoft/trocr-large-handwritten"
headers = {"Authorization": "Bearer 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"}
import time
from array import array
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.speech import SpeechSynthesizer
from azure.cognitiveservices.speech.audio import AudioConfig
from flask import send_file
from io import BytesIO
API_KEY = "a41984e997a74b999e979b08007ec70a"
ENDPOINT = "https://francecentral.api.cognitive.microsoft.com/"
computervision_client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))
import azure.cognitiveservices.speech as speechsdk
speech_config = speechsdk.SpeechConfig("14cee8f9e1fe4d01882ba8a65eb6923f", "francecentral")
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
speech_config.speech_synthesis_voice_name = 'en-US-AnaNeural'
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

# provide login manager with load_user callback
@lm.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))

@login_required
@app.route('/back_index')
def back_index():
    # Access user attributes from the current_user object
    count_activities = update_user_scores(current_user.email)
    activities_count_per_days = get_activities_count_by_day()
    activities_count_per_week = get_activities_count_by_week()
    activities_count_per_month = get_activities_count_by_month()
    recent_activities = get_recent_activities()
    # Render the dashboard template and pass the user data
    return render_template('dashboard.html', user=current_user, count_activities=count_activities, 
                           activities_count_per_days=activities_count_per_days,
                           activities_count_per_week=activities_count_per_week,
                            activities_count_per_month=activities_count_per_month,
                            recent_activities = recent_activities )
# Logout user
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


# Register a new user
@app.route('/auth', methods=['GET', 'POST'])
def register():
    msg = None
    success = False

    if request.method == 'POST':
        if 'signup' in request.form:  # Check if the signup form is submitted
            username = request.form['FullName']
            email = request.form['Email']
            password = request.form['mdp']
            pathology = "ASD"

            # Check if the required fields are not empty
            if username and email and password:
                # Check if a user with the same username or email already exists
                user = Users.query.filter_by(user=username).first()
                user_by_email = Users.query.filter_by(email=email).first()

                if user or user_by_email:
                    msg = 'User already exists!'
                else:
                    pw_hash = bc.generate_password_hash(password)
                    user = Users(username, email, pw_hash,progression=0,Language=0,Intellect=0, Social_Skills=0,pathology='ASD',isPremium=False)
                    user.save()
                    msg = 'Welcome '+username+' to CogniPath'
                    success = True
            else:
                msg = 'Input error'

        elif 'signin' in request.form:  # Check if the signin form is submitted
            email_login = request.form['Email_login']
            password_login = request.form['mdp_login']

            # Check if the required fields are not empty
            if email_login and password_login:
                user = Users.query.filter_by(email=email_login).first()

                if user:
                    if bc.check_password_hash(user.password, password_login):
                        login_user(user)
                        return redirect(url_for('index'))
                    else:
                        msg = "Wrong password. Please try again."
                else:
                    msg = "Unknown user"
            else:
                msg = 'Sign-in failed. Please try again.'

    return render_template('login.html', msg=msg, success=success)


# Authenticate user
@app.route('/login', methods=['GET', 'POST'])
def login():

    # Flask message injected into the page, in case of any errors
    msg = None
    
    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        username = request.form.get('username', '', type=str)
        password = request.form.get('password', '', type=str) 

        # filter User out of database through username
        user = Users.query.filter_by(user=username).first()

        if user:
            
            if bc.check_password_hash(user.password, password):
                login_user(user)
                return redirect(url_for('index'))
            else:
                msg = "Wrong password. Please try again."
        else:
            msg = "Unknown user"

    return render_template( 'login.html', form=form, msg=msg )

# App main route + generic routing
@app.route('/', defaults={'path': 'index'})
@app.route('/<path>')
def index(path):

    #if not current_user.is_authenticated:
    #    return redirect(url_for('login'))

    try:

        return render_template( 'index.html')
    
    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
    except:
        return render_template('page-500.html'), 500

# Return sitemap
@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'sitemap.xml')



@app.route('/homepage', methods=['GET', 'POST'])
def homepage():
    return render_template('index.html')




# @login_required
# @app.route("/generate", methods=["POST"])
# def generate():
#     selected_choice = request.form.get("choice")
#     print("selected item:",selected_choice)
#     activity = "Emotional Recognition"
#     print("selected item:",selected_choice)
#     # Define the payload for the Hugging Face API
#     payload = {
#         "inputs": selected_choice  # Use the selected choice as input
#     }

#     # Send a request to the Hugging Face API
#     try:
#         response = requests.post(API_URL, headers=headers, json=payload)
#         response.raise_for_status()  # Check for any errors in the API response
#         image_bytes = response.content

#         new_activity = Activity(
#             title=selected_choice,
#             input=selected_choice,
#             output="An image",
#             user_email=current_user.email,
#             activity=activity,
#             social_score=10,
#             intellect_score=20,
#             language_score=30,
#             avg_score=20
#         )
#         new_activity.save()
#         update_user_scores(current_user.email)

#         # Generate a unique filename for the image with a .jfif extension
#         filename = "app/static/aa.jfif"

#         # Save the image to a file
#         with open(filename, "wb") as f:
#             f.write(image_bytes)

#         # Pass the filename to the Emotions_Recognition.html template
#         return render_template("Emotions_Recognition.html", image_filename=filename)
    
    # except requests.exceptions.RequestException as e:
    #     # Handle any errors that occur during the API request
    #     return jsonify({"error": str(e)})
    

 
@app.route("/display_image/<filename>")
def display_image(filename):
    return send_file(filename, mimetype='image/jfif')




@app.route('/Essay_correction')
@login_required
def Essay_correction():
    return render_template('EssayCorrection.html')


@app.route('/Text_simplification')
@login_required
def Text_simplification():
    return render_template('Text_simplification.html')

@app.route('/cognipro')
@login_required
def cognipro():
    return render_template('cognipro.html')
@app.route("/generate_text", methods=["POST"])
@login_required
def generate_text():
    try:
        image = request.files["image"]
        if image:
            img_stream = BytesIO(image.read()) # Read the uploaded file as bytes
            img_data = base64.b64encode(img_stream.getvalue()).decode('utf-8')
            read_response = computervision_client.read_in_stream(img_stream, language='en', raw=True)
            read_operation_location = read_response.headers["Operation-Location"]
            operation_id = read_operation_location.split("/")[-1]

            while True:
                read_result = computervision_client.get_read_result(operation_id)
                if read_result.status not in ['notStarted', 'running']:
                    break
                time.sleep(1)

            if read_result.status == OperationStatusCodes.succeeded:
                extracted_text = ''  # Create a variable to store the extracted text
                for text_result in read_result.analyze_result.read_results:
                    for line in text_result.lines:
                        extracted_text += line.text + ' '
            # You can return the extracted text and any other data you want to the template
            return render_template("EssayCorrection.html", generated_text=extracted_text,imgg=img_data)

    except Exception as e:
        return f"An error occurred: {str(e)}"


@app.route("/correct_text", methods=["POST"])
def correct_text():
    activity = "Writing Wizard"
    user_input = request.form["user_input"]

    # Use the user_input as input for ChatGPT
    messages = [
    {
        "role": "user",
        "content": user_input
    },
    {
        "role": "assistant",
        "content": """Great job sharing your work with us! We're here to help you grow as a writer. Please use this format for feedback:

Grade: [Your grade, like A, B, or C]

Strengths:
- [Strength 1]
- [Strength 2]
- [Add more strengths if desired]

Areas for Improvement:
- [Area for Improvement 1]
- [Area for Improvement 2]
- [Add more areas for improvement if desired]

Remember, every great writer started somewhere, and we believe in your potential!"""
    }
]
    try:
        # Make the OpenAI chat completion request
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        assistant_reply = response.choices[0].message["content"]
        print(assistant_reply)

        # Split the response into lines
        lines = assistant_reply.split('\n')

        # Initialize variables to store the grade, strengths, and areas for improvement
        grade = ""
        strengths = ""
        areas_for_improvement = ""

        current_section = None

                # Iterate through the lines to categorize them into grade, strengths, and areas for improvement
        for line in lines:
                if line.startswith("Grade:"):
                    current_section = "Grade"
                    grade += line + '\n'
                elif line.startswith("Strengths:"):
                    current_section = "Strengths"
                    strength_points = line.split("\n")
                    strengths += '\n'.join(point.strip() for point in strength_points if point.strip()) + '\n'
                elif line.startswith("Areas for Improvement:"):
                    current_section = "Areas for Improvement"
                    improvement_points = line.split("\n")
                    areas_for_improvement += '\n'.join(point.strip() for point in improvement_points if point.strip()) + '\n'
                else:
                    if current_section == "Grade":
                        grade += line + '\n'
                    elif current_section == "Strengths":
                        strengths += line + '\n'
                    elif current_section == "Areas for Improvement":
                        areas_for_improvement += line + '\n'
        pros = strengths
        cons=areas_for_improvement  # Replace with the actual areas for improvement
        calculate_score(grade, activity)

    except Exception as e:
        return f"Error generating assistant reply: {str(e)}"
    # Render the HTML template with the variables
    return render_template("EssayCorrection.html", grade=grade, pros=pros, cons=cons, user_input=user_input)
@app.route("/simplify", methods=["GET", "POST"])
def simplify():
    if request.method == "GET":
        return render_template("simplify.html")

   
    user_input = request.form["user_input"]
    generated_text = user_input  # Use the user-provided text as input

    # Use the generated_text as input for ChatGPT
    messages = [
        {
            "role": "system",
            "content": "Simplify the paragraph given please so a 5-year-old can understand it make it like a short story of 6 sentences : "
        },
        {
            "role": "user",
            "content": generated_text  # Use the user-provided text as the user's input
        }
    ]

    # Make the OpenAI chat completion request
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    assistant_reply = response.choices[0].message["content"]

    return render_template("simplify.html", generated_text=generated_text, corrected_text=assistant_reply)
@app.route("/play_voice", methods=["GET"])
def play_voice():
    generated_text = request.args.get("generated_text")

    # Define the path relative to the app's root directory
    output_path = os.path.join(app.root_path, "output.mp3")

    audio_config = AudioConfig(filename=output_path)
    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    synthesizer.speak_text(generated_text)

    return send_file(output_path, as_attachment=True)
def separate_sentences(input_list):
    if len(input_list) == 1:
        # Split the single string in the list into sentences based on the newline character ('\n')
        sentences = input_list[0].split('\n')

        # Remove any leading or trailing whitespace from each sentence
        sentences = [sentence.strip() for sentence in sentences]

        return sentences
    else:
        return []
@app.route('/Emotions')
def Emotions():
    sentences = ['A happy girl in a pink dress', 'A young girl laughing hysterically', 'An old woman feeling nostalgic', 'A boy looking sad and lost']

    # Cache the generated sentences to avoid repeated API calls
    
    prompts = [
    "I want you to generate a 4 random emotion recognition sentences for example : 'A sad lady in red' or 'A happy young girl holding a toy'"
    ]
    output = separate_sentences(generate_emotion_recognition_text(prompts))
    print(output)
    # output = output[0] if output else ""
    # sentences = [sentence.strip() for sentence in output.split('\n') if sentence.strip()]
    # session['sentences'] = sentences

    # Select a random sentence
    correct_emotion = random.choice(output)

    # Fetch the image
    payload = {
        "inputs": correct_emotion
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        image_bytes = response.content
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)})

    # Generate a unique filename for the image with a .jfif extension
    filename = "app/static/aa.jfif"

    # Save the image to a file
    with open(filename, "wb") as f:
        f.write(image_bytes)


    return render_template('Emotions_Recognition.html', sentences=output, correct_emotion=correct_emotion, image_filename=filename)

@app.route('/profile')
def profile():
    return render_template('profile.html')

def generate_emotion_recognition_text(prompts):
    # Initialize the OpenAI API client

    # Initialize an array to store the results
    results = []

    # Generate text for each prompt and append it to the results array
    for prompt in prompts:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=50  # Adjust the max tokens as needed
        )
        result_text = response.choices[0].text.strip()
        results.append(result_text)

    return results


def calculate_score(grade, activity):
    if grade:
        if activity == "Writing Wizard":
            language_score = [ 5 if grade.strip()=="Grade: A" else 4 if grade.strip()=="Grade: B" else 3 if grade.strip()=="Grade: C" else 2]
            social_score = 0
            intellect_score = 0
        elif activity == "Emotion Recognition":
            language_score = 0
            social_score = 0
            intellect_score = 0
        elif activity == "Storify":
            language_score = 2
            social_score = 1
            intellect_score = 0
        
        new_activity = Activity(
            title=activity,
            input=activity,
            output="Output",
            user_email=current_user.email,
            activity=activity,
            social_score=social_score,
            intellect_score=intellect_score,
            language_score=language_score[0],
            avg_score=int(round((social_score+intellect_score+language_score[0])/3,2))
        )
        new_activity.save()
        update_user_scores(current_user.email)


from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

# Load environment variables from .env file
# You should create a .env file and add a line: OPENAI_API_KEY=<your_openai_key>
load_dotenv()

client = OpenAI(api_key="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")

app.secret_key = 'sk1-sSazzapojzafopjafopjazopfjazf05ffzf'

global gpt_response
gpt_response = None
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


gpt_response = None  # Define a global variable to store the GPT response

@app.route('/ask', methods=['POST'])
def ask():
    global gpt_response  # Use the global variable to store the response
    imageUrl = request.form['image_url'].strip()
    promptText = request.form['prompt'].strip()

    if imageUrl and promptText:
        print(promptText + " -> " + imageUrl)
        try:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": promptText},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": imageUrl,
                                    "detail": "high"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1500,
            )
            print(response)
            answer = response.choices[0].message.content.strip()
            gpt_response = answer  # Store the response in the global variable
            return jsonify({'status': 'OK', 'answer': answer})
        except Exception as e:
            print(e)
            return jsonify({'status': 'ERROR', 'answer': str(e)})
    else:
        return jsonify({'status': 'ERROR', 'answer': 'Input Error: Please provide an image and the prompt to ask for GPT4.'})
@app.route('/craft')
def craft():
    global gpt_response  # Access the global variable
    return render_template('craft.html', gpt_response=gpt_response)
   

@app.route('/imagine', methods=['POST'])
def imagine():
    promptText = request.form['prompt'].strip()

    if promptText:
        print(promptText)
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=promptText,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            print(response)
            answer = response.data[0].revised_prompt
            url = response.data[0].url
            return jsonify({'status': 'OK', 'image_url': url, "answer": answer})
        except Exception as e:
            print(e)
            return jsonify({'status': 'ERROR', 'answer': e.message})
    else:
        return jsonify({'status': 'ERROR', 'answer': 'Please provide an image description.'})


@app.route('/recycle', methods=['GET', 'POST'])
def recycle():
    return render_template('recycle.html')

@app.route('/foody', methods=['GET', 'POST'])
def foody():
    return render_template('foody.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        flash('File uploaded successfully!', 'success')
        return redirect(url_for('donation'))

@app.route('/donation')
def donation():
    image_names = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('donation.html', image_names=image_names)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f'{filename} deleted successfully!', 'success')
    else:
        flash(f'{filename} not found!', 'danger')
    return redirect(url_for('donation'))
from flask import Flask, render_template, Response, request
import cv2
import os

video = cv2.VideoCapture(0)
static_folder = os.path.join(app.root_path, 'static')
img_path = os.path.join(static_folder, 'image.jpg')


@app.route('/takeimage', methods=['POST'])
def takeimage():
    global img_path
    _, frame = video.read()
    cv2.imwrite(img_path, frame)
    return Response(status=200)

def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = video.read()
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/preview')
def preview():
    """Preview the captured image."""
    return render_template('recycle.html')
if __name__ == '__main__':
    app.run(debug=True)