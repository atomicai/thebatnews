import datetime
from pathlib import Path

import simplejson as json


def convert_date_to_rfc3339(date: str) -> str:
    """
    Converts a date to RFC3339 format, as Weaviate requires dates to be in RFC3339 format including the time and
    timezone.

    If the provided date string does not contain a time and/or timezone, we use 00:00 as default time
    and UTC as default time zone.

    This method cannot be part of WeaviateDocumentStore, as this would result in a circular import between weaviate.py
    and filter_utils.py.
    """
    parsed_datetime = datetime.fromisoformat(date)
    if parsed_datetime.utcoffset() is None:
        converted_date = parsed_datetime.isoformat() + "Z"
    else:
        converted_date = parsed_datetime.isoformat()

    return converted_date


def is_json(x):
    if issubclass(type(x), Path):
        return True
    try:
        json.dumps(x)
        return True
    except:
        return False


def convert_iob_to_simple_tags(preds, spans, probs):
    contains_named_entity = len([x for x in preds if "B-" in x]) != 0
    simple_tags = []
    merged_spans = []
    tag_probs = []
    open_tag = False
    for pred, span, prob in zip(preds, spans, probs):
        # no entity
        if not ("B-" in pred or "I-" in pred):
            if open_tag:
                # end of one tag
                merged_spans.append(cur_span)
                simple_tags.append(cur_tag)
                tag_probs.append(prob)
                open_tag = False
            continue

        # new span starting
        elif "B-" in pred:
            if open_tag:
                # end of one tag
                merged_spans.append(cur_span)
                simple_tags.append(cur_tag)
                tag_probs.append(prob)
            cur_tag = pred.replace("B-", "")
            cur_span = span
            open_tag = True

        elif "I-" in pred:
            this_tag = pred.replace("I-", "")
            if open_tag and this_tag == cur_tag:
                cur_span = (cur_span[0], span[1])
            elif open_tag:
                # end of one tag
                merged_spans.append(cur_span)
                simple_tags.append(cur_tag)
                tag_probs.append(prob)
                open_tag = False
    if open_tag:
        merged_spans.append(cur_span)
        simple_tags.append(cur_tag)
        tag_probs.append(prob)
        open_tag = False
    if contains_named_entity and len(simple_tags) == 0:
        raise Exception(
            "Predicted Named Entities lost when converting from IOB to simple tags. Please check the format"
            "of the training data adheres to either adheres to IOB2 format or is converted when "
            "read_ner_file() is called."
        )
    return simple_tags, merged_spans, tag_probs


__all__ = ["convert_date_to_rfc3339", "is_json"]
