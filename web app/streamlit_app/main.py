import streamlit as st

from imports import *

# import OpenAI
from openai._client import OpenAI
from open_ai_client import client_chat # contains util function(s) to interact with OpenAI

from plan_generator import generate_plans, select_plan

from face_place_validation import validate_usr_in_plc

#from llm_bot import dummy_bot, echo_bot #contains logic for bot's response


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

# Read Plan Json structure from 'plan.json' file
with open('./plan.json') as f:
    st.session_state.plan = json.load(f)

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
st.image(config['streamlit']['logo'], width=50)

# Set page title
st.title(title)

# ---------------------------
# Start logic

# Use this value to help control streamlit state logic
EMPTY_STATE = "EMPTY_STATE"



# Create OpenAI client
gpt_client = OpenAI() #API_key is automatically read from .env using "load_dotenv" package

# Initialize chat history
if "messages" not in st.session_state:
	st.session_state.messages = []
	msg = "how you feel" # TODO: client_chat(client=gpt_client, user_input="", ass_content="Say greetings and ask if feeling motivated or low today")
	st.session_state.messages.append({
		"role": "assistant", 
		"content": msg
	})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatar[message["role"]]):
        st.markdown(message["content"])

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


    if not 'plans_generated' in st.session_state:
    
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

            st.write("\tWork place:")
            st.write("\t- Name: {0} (Rating: {1})".format(work_place["name"], work_place["rating"]))
            st.write("\t- Maps: {0}".format(work_place["maps_and_photos"]))
            #st.image(str(Path("imgs") / "plc_{0}.jpg".format(work_place["place_id"])))
            st.write("Fun place:")
            st.write("\t- Name: {0} (Rating: {1})".format(fun_place["name"], fun_place["rating"]))
            st.write("\t- Maps: {0}".format(fun_place["maps_and_photos"]))
            #st.image(str(Path("imgs") / "plc_{0}.jpg".format(fun_place["place_id"])))
            st.write("-----")
            
        st.session_state.plans_generated = True
	
	    # Get "Plan Num" value using button group
	    
        def set_plan_num(plan_num):
            st.session_state.plan_num = plan_num
            st.session_state.plan = select_plan(plan_num, st.session_state.plan, "TODO: start_location", st.session_state.rnd_work_plcs, st.session_state.rnd_fun_plcs)
            mlog(st.session_state.plan)
            st.session_state["work_validating"] = True
            st.session_state["work_validated"] = False
            st.session_state["fun_validating"] = False
            st.session_state["fun_validated"] = False

        if st.session_state.plans_generated:
            # Display button group till one of the buttons is clicked
            if 'plan_num' not in st.session_state or (st.session_state.plan_num == None):
                st.session_state.plan_num = None
                if st.button('Plan 1', on_click=set_plan_num, args=(1,)):
                    print("plan num:", st.session_state.plan_num)
                    st.success("Great!. Plan {0} Sounds like a good choice :wink:".format(st.session_state))
                if st.button('Plan 2', on_click=set_plan_num, args=(2,)):
	                st.success("Plan {0} seems interesting :wink:")
	                print("plan num:", st.session_state.plan_num)
            
    if 'work_validating' in st.session_state and st.session_state.work_validating:
            
            # Reset messages queue
            #st.session_state.messages = []
            # clear page
            #placeholder = st.empty()
                
            # Start with selected plan
            msg = client_chat(client=gpt_client, user_input="", ass_content="Let's start the day with work!. You have to visit '{0}'. When you're ready to validate your presence there , please click on the button.".format(st.session_state.plan["places_to_visit"][0]["name"]))
            st.session_state.messages.append({
                "role": "assistant", 
                "content": msg
            })
            
            # as long as not correctly validated keep it visual
            if not st.session_state.work_validated:
            
                if st.button("Upload Photo To Validate"): # handle on click
                    uploaded_file = st.file_uploader("Choose a file")
                    if uploaded_file is not None:
                        mlog(">>>>>>>>>>>", uploaded_file.name)
                        
                #st.session_state.work_validated, msg = validate_usr_in_plc(captured_img_path, val_usr_name, plc_id)
                #mlog("Validation msg:", msg)
                    
            
    
    # React to user input
    if prompt := st.chat_input("Please type 1 or 2"):
        #if prompt := st.chat_input("Please type 1 or 2"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": "Plan {0} seems more interesting".format(st.session_state.plan_num)})
        st.session_state.messages.append({"role": "assistant", "content": "I think this is a great choice :wink:"})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Get bot response    
        response = echo_bot(prompt)
        with st.chat_message("assistant", avatar=config['streamlit']['avatar']):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


    
