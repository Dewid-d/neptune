import requests
import json



class RemoteLogging():
    def __init__(self, webhook: str) -> None:
        self.webhook = webhook
    
    def log(self, msg: str):
        raise NotImplementedError


class DiscordLogging(RemoteLogging):
    def __init__(self, webhook: str, global_rank: int) -> None:
        super().__init__(webhook)
        self.global_rank = global_rank

    def log(self, msg: str):
        if not self.webhook: return
        try:
            if self.global_rank == 0:
                msg = str(msg)
                # create data message
                data = {'content':msg}
                # post to the message to the webhook
                requests.post(self.webhook, data=json.dumps(data), headers={"Content-Type": "application/json"})
        except Exception as e:
            print(f'Discord Message not sent. Error: {e}')


class SlackLogging(RemoteLogging):
    def __init__(self, webhook: str) -> None:
        super().__init__(webhook)

    def log(self, msg: str):
        # create data message
        data = {'text':msg}
        # post to the message to the webhook
        requests.post(self.webhook, data=json.dumps(data), headers={"Content-Type": "application/json"})