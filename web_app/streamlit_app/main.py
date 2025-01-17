import streamlit as st

from imports import *

# import OpenAI
from openai._client import OpenAI
from open_ai_client import client_chat # contains util function(s) to interact with OpenAI

from plan_generator import generate_plans, select_plan

from face_place_validation import validate_usr_in_plc

def mlog(msg):
	"""
	Simple logging to console function
	Parameters
    	----------
    	msg: str
    		The message to be logged to console
	"""
		
	print("[log] - {0}.\n".format(msg))
	return None


	
# Load .env vars
load_dotenv()

# run as demo. Aschaffenburg Central Station as starting search point. Plan 2 will be the one stored in plan_demo.json (work: library, fun: eat at hotel restaurant)
st.session_state.demo = True

# Read Plan Json structure from 'plan.json' file
with open('./plan.json') as f:
    st.session_state.plan_template = json.load(f)
    
# Set default user name
st.session_state.user_name = "Omar"

# Read config yaml file
with open('./config.yml', 'r') as file:
    config = yaml.safe_load(file)
    
title = config['streamlit']['title']
avatar = {
    'user': None,
    'assistant': config['streamlit']['avatar']
}

# Set page config
st.set_page_config(
    page_title=config['streamlit']['tab_title'], 
    page_icon=config['streamlit']['page_icon'], 
    )

# Set sidebar
st.sidebar.title("About")
st.sidebar.info(config['streamlit']['about'])

# Set logo
st.image(config['streamlit']['logo'], width=128)

# Set page title
st.title(title)

# ---------------------------
# Start logic

# Create OpenAI client
gpt_client = OpenAI() #API_key is automatically read from .env using "load_dotenv" package

if not 'plan_num' in st.session_state:
    msg = client_chat(client=gpt_client, user_input="", ass_content="Say greetings and ask if feeling motivated or low today")
    st.write(msg)


# Get "Motivated" or "Not" response using a button group
    def set_user_motivated_val(val):
        st.session_state.user_motivated = val

    # Display button group till one of the buttons is clicked
    if 'user_motivated' not in st.session_state or (st.session_state.user_motivated == None):
        st.session_state.user_motivated = None
        if st.button('Motivated! 😁', on_click=set_user_motivated_val, args=(True,)):
	        st.success("Great!. Then let's do our best :wink:")
        if st.button('A Little Bit Down 😔', on_click=set_user_motivated_val, args=(False,)):
	        st.success("No worries. I will create you an encouraging plan :wink:")

    if 'user_motivated' in st.session_state and st.session_state.user_motivated in [True, False]:
        if st.session_state.user_motivated == True:
	        st.success("Let's do our best :wink:")
        elif st.session_state.user_motivated == False:
	        st.success("Keep it up. I've got an encouraging plan :wink:")

# if user mood state is captured
if 'user_motivated' in st.session_state and not st.session_state.user_motivated is None:

    if not 'plan' in st.session_state:

        st.session_state.plan = None
    
        st.write("I will generate 2 plans for today so you can pick the one you like :wink:")
        # Generate 2 plans using Google Places
        rnd_work_plcs, rnd_fun_plcs = generate_plans(st.session_state.user_motivated)
        st.session_state["rnd_work_plcs"] = rnd_work_plcs
        st.session_state["rnd_fun_plcs"] = rnd_fun_plcs

        # Display plans
        # Plan 1
        for i in range(1,3, 1):
            plan_num = i
            plan_idx = i -1
            work_place = rnd_work_plcs[plan_idx]
            fun_place = rnd_fun_plcs[plan_idx]

            st.write(f"**Plan {plan_num}:**")

            st.write("\t> Work place:")
            st.write("\t\t- Name: {0} (Rating: {1})".format(work_place["name"], work_place["rating"]))
            st.write("\t\t- Google Maps: [Click here]({0})".format(work_place["maps_and_photos"]))

            
            st.write("\t> Fun place:")
            st.write("\t\t- Name: {0} (Rating: {1})".format(fun_place["name"], fun_place["rating"]))
            st.write("\t\t- Google Maps: [Click here]({0})".format(fun_place["maps_and_photos"]))
            
            #st.image(str(Path("imgs") / "plc_{0}.jpg".format(fun_place["place_id"])))
            st.write("-----")
            
        #st.session_state.plans_generated = True
	
	    # Get "Plan Num" value using button group
	    
        def set_plan_num(plan_num):
            st.session_state.plan_num = plan_num
            # Set plan session state
            
            # Remove previously saved plans' places images
            for f_path in Path("imgs").glob("plc_*.jpg"):
                os.remove(str(f_path))
            
            if st.session_state.demo:

                st.warning("Running in DEMO MODE")
                
                # Copy places images from imgs/demo_places to imgs/ since we don'r retrieve them in demo mode
                demo_imgs_dir = str(Path("imgs") / "demo_places")
                imgs_dir = str(Path("imgs"))
                shutil.copytree(demo_imgs_dir, imgs_dir, dirs_exist_ok=True) # Copy all files in demo_places to destination
                
                # Read demo plan json structure from 'plan_demo.json'
                with open('./plan_demo.json') as f:
                    demo_plan = json.load(f)
                    # Replace empty date by today's date
                    demo_plan["date"] = datetime.today().strftime('%d.%m.%Y')
                    st.session_state.plan = demo_plan
            else:
                st.session_state.plan = select_plan(plan_num, st.session_state.plan_template, "TODO: allow different start_locations", st.session_state.rnd_work_plcs, st.session_state.rnd_fun_plcs)
            
            st.session_state["work_validating"] = True
            st.session_state["work_validated"] = False
            #st.session_state["fun_validating"] = False
            #st.session_state["fun_validated"] = False

        #if st.session_state.plans_generated:
        # Display button group till one of the buttons is clicked
        if 'plan_num' not in st.session_state:
            st.session_state.plan_num = None
            if st.button('Plan 1', on_click=set_plan_num, args=(1,)):
                print("plan num:", st.session_state.plan_num)
            if st.button('Plan 2', on_click=set_plan_num, args=(2,)):
                print("plan num:", st.session_state.plan_num)
	                
    # if plan is selected
    else:
    
        def plc_validate(validating_type):
            """
            Validates work or fun place using uploaded captured photo of user and the image of place from Google map.
            
            Parameters:
            ----------------
            validating_type: str
                A value to indicate if we are validating the work ('work_validating') place or the fun ('fun_validating') flace
            """
            
            if validating_type == 'work_validating':
                st.session_state.validate_place = st.session_state.plan["places_to_visit"][0]
                session_validated_key = 'work_validated'
            elif validating_type == 'fun_validating':
                st.session_state.validate_place = st.session_state.plan["places_to_visit"][1]
                session_validated_key = 'fun_validated'
            
            msg = client_chat(client=gpt_client, user_input="", ass_content="Let's visit '{0}'!. When you're ready to validate your presence there , please click on the button.".format(st.session_state.validate_place["name"]))
            st.write(msg)
            st.write("Please upload a photo of your self so the background looks similar to this:")
            st.image(str(Path("imgs") / "plc_{0}.jpg".format(st.session_state.validate_place["place_id"])))
            
            # as long as not correctly validated keep it visual
            if not st.session_state[session_validated_key]:
            
                uploadbtn = st.button("Upload Photo To Validate {0} Place".format(validating_type.split("_")[0].capitalize()))
                
                if "uploadbtn_state" not in st.session_state:
                    st.session_state.uploadbtn_state = False
                    
                if uploadbtn or st.session_state.uploadbtn_state:
                    st.session_state.uploadbtn_state = True
                    
                    st.session_state.uploaded_img = st.file_uploader("Choose a file")
                    
                    if st.session_state.uploaded_img is not None:
                        # 1- Save uploaded photo
                        captured_img_path = Path("imgs") / "captured" / "user_upload.jpg"
                        with open(captured_img_path, mode='wb') as w:
                            w.write(st.session_state.uploaded_img.getvalue())
                            
                        # 2- validate face and place.
                        st.session_state[session_validated_key], msg = validate_usr_in_plc(captured_img_path, st.session_state.user_name, st.session_state.validate_place["place_id"])
                        if st.session_state[session_validated_key]:
                            st.write("🥳🥳 " + msg)
                            st.session_state.plan["gained_coins"] += 50
                            st.write("💎 You have just got +50 BONUS 💎")
                            
                            st.session_state[session_validated_key] = True # either 'work_validated' or 'fun_validated'
                            
                            # Switch to fun place validating if validating work place
                            if st.session_state["work_validating"]:
                                st.session_state.plan["places_to_visit"][0]["validated"] = True
                                st.session_state.work_validating = False
                                
                                # Create fun session state vars
                                st.session_state.fun_validating = True
                                st.session_state.fun_validated = False
                                
                            elif st.session_state["fun_validating"]:
                                st.session_state.plan["places_to_visit"][1]["validated"] = True
                                st.session_state.writing_memo = True
                                
                        # If validation fails    
                        elif not st.session_state[session_validated_key]:
                            st.write("⚠️ " + msg)
                            # Allow processing new upload
                            st.session_state.uploaded_img = None

            return None
        
        # Start places validation logic
        if st.session_state.work_validating and not st.session_state.work_validated: # validating work place which is still not vlidated
        
            # clear page
            st.empty()
            plc_validate('work_validating')
            
        if 'fun_validating' in st.session_state and not st.session_state.fun_validated: # validating fun place which is still not vlidated
        
            btn_fun = st.button("It's fun time! :wink:")
            
            if not 'btn_fun_state' in st.session_state:
                st.session_state.btn_fun_state = False
                # del / reset previous photo upload session state vars
                del st.session_state["uploadbtn_state"]
                st.session_state.uploaded_img = None
                # clear page
                st.empty()
                
            if btn_fun or st.session_state.btn_fun_state:
                st.session_state.btn_fun_state = True
                plc_validate('fun_validating')
                
            
        # If places are validated
        if 'writing_memo' in st.session_state:
        
            btn_memo = st.button("Time to finish the day :wink:")
            
            if not 'btn_memo_state' in st.session_state:
                st.session_state.btn_memo_state = False
                
            if btn_memo or st.session_state.btn_memo_state:
                st.session_state.btn_memo_state = True
        
                if not 'btn_memo_clicked' in st.session_state:
                    # clear page
                    st.empty()
                    
                    msg = "📝 " + client_chat(client=gpt_client, user_input="", ass_content="You must be proud of your self! How about writing a short memo about what you achieved today? This will be saved with today's plan and gives you extra +30 BONUS")
                    st.write(msg + " :wink:")
                    
                    memo = st.text_input("Today's Memo", "")
                    if st.button('Save and End'):
                        st.session_state.btn_memo_clicked = True
                        if not memo == "":
                            st.session_state.plan["memo"] = memo
                            st.session_state.plan["gained_coins"] += 30
                            st.write("💎 You have just got +30 BONUS 💎")
                        else:
                            st.write("Empty memo. no extra bonus for today 😔. Maybe you write a short note next time? :wink:")
                            
                        st.write("🥳 Great day! 🥳")
                        
                        # Debugging
                        # Save as JSON
                        today_plan_path = str(Path("daily_plans") / "plan_{0}.json".format(st.session_state.plan["date"]))
                        with open(str(today_plan_path), 'w') as f:
                            json.dump(st.session_state.plan, f)
                            
                            


