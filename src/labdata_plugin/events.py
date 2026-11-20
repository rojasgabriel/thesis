"""Task event names of each session."""

from ._base import dj, rojasbowe_schema


@rojasbowe_schema
class EventMapping(dj.Lookup):
    definition = """
    stream_name                          : varchar(54)
    event_name                           : varchar(54)
    ---
    event_role                           : varchar(54)
    """
    default_contents = [
        {"stream_name": "nidq", "event_name": "ai0", "event_role": "visual_stim"},
        {"stream_name": "nidq", "event_name": "0", "event_role": "trig"},
        {"stream_name": "nidq", "event_name": "1", "event_role": "frames"},
        {"stream_name": "nidq", "event_name": "2", "event_role": "trial_start"},
        {"stream_name": "nidq", "event_name": "3", "event_role": "left_port"},
        {"stream_name": "nidq", "event_name": "4", "event_role": "center_port"},
        {"stream_name": "nidq", "event_name": "5", "event_role": "right_port"},
        {"stream_name": "obx", "event_name": "io0", "event_role": "visual_stim"},
        {"stream_name": "obx", "event_name": "io1", "event_role": "audio_stim"},
        {"stream_name": "obx", "event_name": "io2", "event_role": "trial_start"},
        {"stream_name": "obx", "event_name": "io3", "event_role": "frames"},
        {"stream_name": "obx", "event_name": "io4", "event_role": "left_port"},
        {"stream_name": "obx", "event_name": "io5", "event_role": "center_port"},
        {"stream_name": "obx", "event_name": "io6", "event_role": "right_port"},
    ]
