from imports import *
# Load .env vars
load_dotenv()	

import googlemaps # Github: https://github.com/googlemaps/google-maps-services-python/tree/master

#init python-google-maps-api client
try:
    del gmaps
except:
    pass
gmaps = googlemaps.Client(os.environ.get("GOOGLE_PLACES_API_KEY"))

# Simply convert a dict to a class 
class Dict2Class(object): 
      
    def __init__(self, dct): 
          
        for key in dct: 
            setattr(self, key, dct[key]) 

def load_plc_srch_config():
    """Loads Google Places search config from 'plc_srch_config.yml'

    Returns
    -------
    Class
        a config class with attrs as settings to be passed to Google Places API and to filter results
    """
    
    with open('plc_srch_config.yml') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)
        # turn config dict into class for ease of use
        cls = Dict2Class(config["DEFAULT"])
    return cls


def filter_places(gmaps_places_res, max_n_plcs, min_rating=0):
    """Filters Google Places search's results according to the passed criteria

    Parameters
    ----------
    gmaps_places_res : list
        The list of Google Places to filter according to the criteria
    max_n_plcs: int
        The maximum number of filtered places to be returned if matching filtering criteria
    min_rating: int, optional
        The minimum acceptable rating for a place on Google Maps to be considiered as a match

    Returns
    -------
    list
        a list of dicts each represents a filtered Google Place
    """
    
    lst_plcs = []
    for place in gmaps_places_res["results"]:
        place_id = place["place_id"]
        name = place["name"]
        if "rating" in place:
            rating = place["rating"]
        else:
            rating = "NA"
        if rating == "NA" or rating < min_rating:
            continue # Exclude this place
        try:
            maps_and_photos = place["photos"][0]["html_attributions"][0].replace('<a href="', '').split('">')[0]
            photo_reference = place["photos"][0]["photo_reference"]
            
        except:
            maps_and_photos = "NA"
            continue # Only keep places with photos to be able to validate user face next to them later
                
        lst_plcs.append(
            {
                "place_id": place_id,
                "name": name,
                "rating": rating,
                "maps_and_photos": maps_and_photos,
                "photo_reference": photo_reference
            }
        )

    # Filter results to max_n_places
    return random.sample(lst_plcs, max_n_plcs)


def generate_work_plcs(user_motivated):
    """Generates Google Places sutiable for work and matching filter criteria

    Parameters
    ----------
    user_motivated : bool
        A flag used to determine the type of work places that should be included in the plan.

    Returns
    -------
    list
        a list of dicts each represents a filtered Google Place
    """
    
    # Load search config as a Class
    cls_srch_conf = load_plc_srch_config()
    # Shorten it for ease of use
    conf = cls_srch_conf
    
    work_plcs = gmaps.places_nearby(
        location = (conf.start_location["lat"], conf.start_location["long"]),
        radius = conf.search_radius,
        type = conf.motivated_work_plc_types if user_motivated else conf.lazy_work_plc_types,
        language = "en")

    # Debugging
    #print("Found {0} work places.".format(len(work_plcs["results"])))

    # Pick random 2 matching the criteria
    rnd_work_plcs = filter_places(work_plcs, max_n_plcs = conf.max_n_plcs, min_rating = conf.min_rating)

    return rnd_work_plcs


def generate_fun_plcs():
    """Generates Google Places sutiable for having fun and matching filter criteria

    Returns
    -------
    list
        a list of dicts each represents a filtered Google Place
    """
    
    # Load search config as a Class
    cls_srch_conf = load_plc_srch_config()
    # Shorten it for ease of use
    conf = cls_srch_conf
    
    fun_plcs = gmaps.places_nearby(
        location = (conf.start_location["lat"], conf.start_location["long"]),
        radius = conf.search_radius,
        type = conf.fun_plc_types,
        language = "en")

    # Debugging
    #print("Found {0} fun places.".format(len(fun_plcs["results"])))

    # Pick random 2 matching the criteria
    rnd_fun_plcs = filter_places(fun_plcs, max_n_plcs = conf.max_n_plcs, min_rating = conf.min_rating)

    return rnd_fun_plcs


def generate_plans(user_motivated):
    """Generate two plans for today using Google Places API, so the user can choose one later

    Parameters
    ----------
    user_motivated: bool
	    a value that indicates the user mood state today. This will be used to search work places carefully (e.g. library vs cafeteria)

    Returns
    -------
    rnd_work_plcs: list
        Two randomly selected work places out of all returned places that match search criteria
    rnd_fun_plcs: list
        Two randomly selected fun places out of all returned places that match search criteria
    """

    # Init python-google-maps-api client
    try:
        del gmaps
    except:
        pass
    gmaps = googlemaps.Client(os.environ.get("GOOGLE_PLACES_API_KEY"))

    # Load search config as a Class
    cls_srch_conf = load_plc_srch_config()
    # Shorten it for ease of use
    conf = cls_srch_conf

    # We should get 2 work places and 2 fun places to allow the user to select 1 out of 2 plans, each exists from 1 work plc and 1 fun plc
    rnd_work_plcs = generate_work_plcs(user_motivated)
    rnd_fun_plcs = generate_fun_plcs()

    return rnd_work_plcs, rnd_fun_plcs


def select_plan(plan_num, plan, start_location, rnd_work_plcs, rnd_fun_plcs):
    """Selects one of the 2 generated plans and populate the values to the plan JSON.

    Parameters
    ----------
    plan_num : int
        The number of plan to choose. Can be either (1) or (2)
    plan: dict
        The plan structure to be populated with selected plan values
    start_location: str
        The search start location
    rnd_work_plcs: list
        The random work places generated according to search and filter criteria
    rnd_fun_plcs: list
        The random fun places generated according to search and filter criteria

    Returns
    -------
    dict
        Plan structure populated with values from selected plan
    """
    
    # Use 0-based counting instead of 1-based
    plan_num = plan_num - 1
    
    work_plc = rnd_work_plcs[plan_num]
    fun_plc = rnd_fun_plcs[plan_num]
    
    plan["date"] = datetime.today().strftime('%d.%m.%Y')
    plan["start_location"] = start_location
    # work place
    plan["places_to_visit"][0] = {
        "place_id": work_plc["place_id"],
        "name": work_plc["name"],
        "rating": work_plc["rating"],
        "maps_and_photos": work_plc["maps_and_photos"],
        "photo_reference": work_plc["photo_reference"],
        "memo": "",
        "validated": False
    }
    # fun place
    plan["places_to_visit"][1] = {
        "place_id": fun_plc["place_id"],
        "name": fun_plc["name"],
        "rating": fun_plc["rating"],
        "maps_and_photos": fun_plc["maps_and_photos"],
        "photo_reference": fun_plc["photo_reference"],
        "memo": "",
        "validated": False
    }
    
    # Save photos before their reference times out
    max_width = 400
    work_plc_img_URL = "https://maps.googleapis.com/maps/api/place/photo?maxwidth={0}&photo_reference={1}&key={2}".format(max_width, work_plc["photo_reference"], os.environ.get("GOOGLE_PLACES_API_KEY"))
    urllib.request.urlretrieve(work_plc_img_URL, Path("imgs") / "plc_{0}.jpg".format(work_plc["place_id"]))
    work_plc_img_URL = "https://maps.googleapis.com/maps/api/place/photo?maxwidth={0}&photo_reference={1}&key={2}".format(max_width, fun_plc["photo_reference"], os.environ.get("GOOGLE_PLACES_API_KEY"))
    urllib.request.urlretrieve(work_plc_img_URL, Path("imgs") / "plc_{0}.jpg".format(fun_plc["place_id"]))
    print("Photos / images of work and fun places saved locally.")
    
    return plan
