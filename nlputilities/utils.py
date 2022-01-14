import json
import logging
import os

logger = logging.getLogger(__name__)


def safely_load_json(json_path):
    """
    Tries to load and deserialize JSON, returns None when not successful

    :param json_path: path of the JSON file to loaded
    :type json_path: str
    :return: the loaded json if successful, else None
    :rtype: dict
    """
    if json_path and os.path.isfile(json_path):
        logger.debug("Trying to load file at %s", json_path)

        try:
            with open(json_path, "r") as fp:
                logger.debug("Loaded file at %s", json_path)
                return json.load(fp)
        except json.JSONDecodeError as e:
            logger.error("Unable to deserialize %s", e)
            return None
    elif not json_path:
        logger.error("Got None as json path skipping loading config file")
        return None
    else:
        logger.error("%s is not a file", json_path)
        return None