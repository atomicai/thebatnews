import simplejson as json
import logging
from flask import jsonify, request

logger = logging.getLogger(__name__)


def search():
    data = request.get_data(parse_form_data=True).decode("utf-8-sig")
    data = json.loads(data)
    return jsonify({"hello": "world"})

