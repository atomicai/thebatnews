import simplejson as json
import logging
from pathlib import Path
import os
from flask import jsonify, request, session, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import polars as pl
import uuid
# import pyarrow.parquet as pq
import random_name
from icecream import ic
from more_itertools import chunked
from thebatnews.running import INFRunner

logger = logging.getLogger(__name__)

cache_dir = Path(os.getcwd()) / ".cache"

runner = INFRunner.load(Path(os.getcwd()) / "weights", gpu=True, num_processes=0, batch_size=16)


def search():
    data = request.get_data(parse_form_data=True).decode("utf-8-sig")
    data = json.loads(data)
    return jsonify({"hello": "world"})


def upload():
    global runner
    response = {}
    logger.info("welcome to upload`")
    xf = request.files["file"]
    prefixname = random_name.generate_name()
    filename = secure_filename(prefixname + xf.filename)
    ic(f"{filename}")
    if "uid" not in session.keys():
        uid = uuid.uuid4()
        session["uid"] = str(uid)
    else:
        uid = session["uid"]

    if not (cache_dir / str(uid)).exists():
        (cache_dir / str(uid)).mkdir(parents=True, exist_ok=True)
    destination = cache_dir / str(uid) / filename
    fname, fpath = Path(filename), Path(destination)
    df, columns = None, None
    is_suffix_ok, is_file_corrupted = True, False
    if fpath.suffix not in (".xlsx", ".excel", ".csv"):
        is_suffix_ok = False
    else:
        try:
            xf.save(str(destination))
            if fpath.suffix in (".xlsx", ".excel"):
                df = pl.read_excel(destination)
            else:
                df = pl.read_csv(destination)
        except:
            is_file_corrupted = True
        else:
            is_file_corrupted = False

    if is_suffix_ok and not is_file_corrupted:  # is suffix_ok is also false
        docs = df.select("text").to_series().to_list()
        channel_id = df.select("channel_id").to_series().to_list()
        response = []
        for chunk in chunked(docs, n=1024):
            _docs = [{"text": t} for t in chunk]
            labels = []
            for ans in runner.inference_from_dicts(_docs):
                for label in ans["predictions"]:
                    labels.append(label["label"])
            response.extend(labels)
        final_docs =[{"text": _d, "label": _l, "channel_id": _ch} for _d, _l, _ch in zip(docs, response, channel_id)]
        
        arr = pl.from_dicts(final_docs)
        
        fname = Path(Path(filename).stem + ".csv")
        destination = cache_dir / str(uid) / str(fname)
        arr.write_csv(fname)
        return send_file(str(destination), as_attachment=True)


def rock_n_roll():
    """
    This route will scan the file and perform the "lazy" way to push it to the database

    sets the `email` | `text` | `date` columns
    """
    data = request.get_json()
    #
    uid = session["uid"]
    filename = Path(session["filename"])
    text_column, datetime_column, num_clusters = (
        data.get("text", None),
        data.get("datetime", None),
        data.get("num_clusters", None),
    )
    is_text_ok, is_date_ok, is_num_clusters_ok = False, False, False
    df = pl.scan_parquet(cache_dir / uid / f"{filename.stem}.parquet")
    if text_column is not None and text_column in df.columns:
        session["text"] = text_column
        is_text_ok = True
    if datetime_column is not None and datetime_column in df.columns:
        session["datetime"] = datetime_column
        is_date_ok = True

    if num_clusters is not None and int(num_clusters) > 0 and int(num_clusters) < 30:
        session["num_clusters"] = int(num_clusters)
        is_num_clusters_ok = True
    else:
        logger.info(
            f"The number of clusters specified {str(num_clusters)} is not in a reasonable range. Setting default to {str(7)}"
        )
        session["num_clusters"] = 7

    session["email"] = data.get("email", None)

    if is_date_ok and is_text_ok:
        if is_num_clusters_ok:
            return jsonify({"success": "In progress to the moon 🚀"})
        else:
            return jsonify({"success": "In progress to the moon 🚀 with d"})
    else:
        return jsonify({"Error": "Back to earth ♁. Fix the column name(s) 🔨"})
