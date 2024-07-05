# RW-Buddy
![RW-Buddy Logo](/web_app/streamlit_app/images/logo.png "RW-Buddy Logo")
An AI Agent That Brings Joy Back Into Remote-Workers Life by suggesting a daily plan that contains places to visit for working and having fun.

## Features:
- Day plan generating using **Google Places API** and according to search criteria. Two plans are generated so the user can use.
- Day plan is generated according to the **activity state of the user** (i.e. motivated -> e.g. work at library / a little bit down -> e.g. work at a restaurant).
- **Validating places visits** by matching a photo taken by the user and the photo provided by the place by Google Maps.
- **Human-like generated messages** using OpenAI and GPT with temperaute > 0.
- Collecting bonus points. Just for fun 😉

## OpenCV:
- Frontalface Cascade Classifier: to detect faces in images.
- LOCAL BINARY PATTERNS HISTOGRAM (LBPH) Face Recognizer: to be trained on faces dataset for later face id detection
- cv.findHomography(): to find feature matches between captured photos and ones provided by Google Maps in order to validate that the photo is taken in the right place.

## Web App:
All the code is wrapped in a full-stack [Streamlit](https://streamlit.io/) App.

### Local Run - Without Docker:
1. Create conda env using the web_app/environment.yml.
2. Activate conda env using: conda activate rw-buddy.
3. Create an .env file at web_app/streamlit_app/.env and fill with your credentials.
4. Navigate to web_app/streamlit_app/ then run streamlit run main.py. A tab should open automatically in your web browser.

### Local Run - With Docker:
1. Install Docker if not installed.
2. Navigate to web_app/
3. Type "docker run" and hit Enter.

### Deployment
The app is ready to be deployed to Azure and uses [this Micosoft provided repo](https://github.com/microsoft/azure-streamlit-chatbot) as an initial template.

## Demo Video:
Would like to watch in action 🤩? Click [here](https://youtu.be/fnhrjF15kyQ) for a Youtube video demo ▶️ of the app 😊.

## Screenshot:
![RW-Buddy Demo Screenshot](/demo/screenshot.png "RW-Buddy Demo Screenshot")

