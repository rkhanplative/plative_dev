import re
from datetime import datetime, timedelta
import pytz
from plaitotools import PlaitoTools
import json
from slack_sdk.errors import SlackApiError

############################################
# class that handles all the slack interactions
# it is instantiated with the slack client, the ai query and the model
# it has functions to interact with the slack client, send messages, and get recaps
# it also has a function to load the prompts from the config file

class SlackHelper:
    ############################################
    # slack_client refers to the webclient instatiated with the slackbot token in the main class
    # ai_query refers to the method instatiated in the main class that is used to query the ai (basicChatCompletion or longQuery)
    # model refers to the model instatiated in the main class that is used to query the ai (gpt4, cld, icl... etc.)
    
    def __init__(self, slack_client, ai_query, model):
        
        # load the slack client and ai query and model from the init parameters
        self.client = slack_client
        self.ai_query = ai_query
        self.model = model
        
        # load the prompts from the config file
        self.load_config()

    ############################################
    # function loads the prompts from the config file
    def load_config(self):
        config_file = "./plaito_config.json"
        with open(config_file, "r") as f:
            config = json.load(f)
            self.prompts = config['prompts']
            f.close()        
    
    ############################################
    # function returns a recap of a channel over the last 24 hours
    # if the user is not a member of the channel, it returns a message saying so
    # if the channel is not found, it returns a message saying so       
    def get_recap(self, channel_id, user_id):
        
        try:
            
            # check if the user is a member of the channel
            if user_id not in self.client.conversations_members(channel=channel_id)['members']:
                return "You are not a member of this channel"
            
        except SlackApiError as e:
            PlaitoTools.print_debug(f'INFO: ERROR: Error retrieving conversation members: {e}')
            return 'Error retrieving conversation members'
        
        try:
            
            # get json of conversation history using slack api
            result = self.client.conversations_history(channel=channel_id)
            
        except SlackApiError as e:
            
            PlaitoTools.print_debug(f'INFO: ERROR: Error retrieving conversation history: {e}')
            return 'Error retrieving conversation history'
        
        # if the channel is not found, return a message saying so
        if result['ok'] == False:
            return 'Channel Not Found'
        
        conversation_history = {
            "messages":[]
        }
        
        # geneerate a json that contains the conversation history of the channel
        for message in result["messages"]:
            # if the parent message is from the last 24 hours, add it to the conversation history
            if self.is_within_last_24_hours(message):
                conversation_history['messages'].append({
                    'message':self.process_message(message),
                    'threaded_messages':self.get_thread(channel_id,message['ts']) if message.get('thread_ts') is not None else None,
                    })
        
        # if there are no new messages, return a message saying so, otherwise return converation_history_summary
        conversation_history_summary = f'No New Messages in This Channel to Summarize' if len(conversation_history['messages']) == 0 else self.ai_query(self.prompts['recap'],json.dumps(conversation_history,indent=4),self.model)
        return conversation_history_summary 
    
    ############################################
    # function returns a json of threaded messages given a channel and a timestamp
    def get_thread(self, channel_id, ts):
        try:
            thread_json = self.client.conversations_replies(channel=channel_id,ts=ts)
        except SlackApiError as e:
            PlaitoTools.print_debug(f'INFO: ERROR: Error retrieving conversation replies: {e}')
            return 'Error retrieving conversation replies'
        
        thread_messages = []
        for message in thread_json["messages"]:
            thread_messages.append({'threaded_message':self.process_message(message)})
        return thread_messages
    
    ############################################
    # function processes a message and replaces user_ids with their names
    def process_message(self, message):
        text = message["text"] if message.get('text') != '' else message.get('attachments')[0]['text']
        sender = message["user"]
        message_str = f'<@{sender}> sent {text}'
        
        # replace all user_ids in the message with their names
        user_ids = list(set(re.findall(r"<@(.*?)>", message_str)))
        for user_id in user_ids:
            
            try:
                user_info = self.client.users_info(user=user_id)['user']
            except SlackApiError as e:
                PlaitoTools.print_debug(f'INFO: ERROR: Error retrieving user info: {e}')
                return f'Error retrieving user info for {user_id}'
            
            name = user_info['name'] 
            real_name = user_info.get('real_name')
            message_str = message_str.replace(user_id,name if real_name is None else real_name)
        return message_str
    
    ############################################
    # functions checks whether message was sent in the last 24 hours
    def is_within_last_24_hours(self, message):
        # Parse the timestamp string into a datetime object
        timestamp = datetime.fromtimestamp(float(message['ts']))

        # Convert the datetime object to the local timezone
        local_tz = pytz.timezone('America/New_York')  # Change this to your local timezone
        local_timestamp = timestamp.astimezone(local_tz)

        # Get the current time in the local timezone
        current_time = datetime.now(local_tz)

        # Calculate the time difference between the current time and the given timestamp
        time_difference = current_time - local_timestamp

        # Check if the time difference is within the last 24 hours
        return time_difference <= timedelta(days=1)