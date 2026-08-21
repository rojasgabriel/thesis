import datajoint as dj
from labdata.schema import get_user_schema

rojasbowe_schema = get_user_schema()


@rojasbowe_schema
class EventMapping(dj.Lookup):
    definition = """
    stream_name                          : varchar(54)
    event_name                           : varchar(54)
    ---
    event_role                           : varchar(54)
    """
    contents = [
        ("nidq", "ai0", "visual_stim"),
        ("nidq", "0", "trig"),
        ("nidq", "1", "frames"),
        ("nidq", "2", "trial_start"),
        ("nidq", "3", "left_port"),
        ("nidq", "4", "center_port"),
        ("nidq", "5", "right_port"),
        ("obx", "io0", "visual_stim"),
        ("obx", "io2", "trial_start"),
        ("obx", "io3", "frames"),
        ("obx", "io4", "left_port"),
        ("obx", "io5", "center_port"),
        ("obx", "io6", "right_port"),
    ]
