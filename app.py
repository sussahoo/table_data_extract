from PIL import Image, ImageEnhance, ImageOps
import string
from collections import Counter
from itertools import tee, count
import pytesseract
from pytesseract import Output
import json
import pandas as pd

# import matplotlib.pyplot as plt
import cv2
import numpy as np
from transformers import DetrFeatureExtractor
from transformers import TableTransformerForObjectDetection
import torch
import gradio as gr


def plot_results_detection(
    model, image, prob, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax
):
    plt.imshow(image)
    ax = plt.gca()

    for p, (xmin, ymin, xmax, ymax) in zip(prob, bboxes_scaled.tolist()):
        cl = p.argmax()
        xmin, ymin, xmax, ymax = (
            xmin - delta_xmin,
            ymin - delta_ymin,
            xmax + delta_xmax,
            ymax + delta_ymax,
        )
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color="red",
                linewidth=3,
            )
        )
        text = f"{model.config.id2label[cl.item()]}: {p[cl]:0.2f}"
        ax.text(
            xmin - 20,
            ymin - 50,
            text,
            fontsize=10,
            bbox=dict(facecolor="yellow", alpha=0.5),
        )
    plt.axis("off")


def crop_tables(pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
    """
    crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates
    """
    cropped_img_list = []

    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

        xmin, ymin, xmax, ymax = (
            xmin - delta_xmin,
            ymin - delta_ymin,
            xmax + delta_xmax,
            ymax + delta_ymax,
        )
        cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
        cropped_img_list.append(cropped_img)
    return cropped_img_list


def add_padding(pil_img, top, right, bottom, left, color=(255, 255, 255)):
    """
    Image padding as part of TSR pre-processing to prevent missing table edges
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def table_detector(image, THRESHOLD_PROBA):
    """
    Table detection using DEtect-object TRansformer pre-trained on 1 million tables
    """

    feature_extractor = DetrFeatureExtractor(do_resize=True, size=800, max_size=800)
    encoding = feature_extractor(image, return_tensors="pt")

    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection"
    )

    with torch.no_grad():
        outputs = model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]["boxes"][keep]

    return (model, probas[keep], bboxes_scaled)


def table_struct_recog(image, THRESHOLD_PROBA):
    """
    Table structure recognition using DEtect-object TRansformer pre-trained on 1 million tables
    """

    feature_extractor = DetrFeatureExtractor(do_resize=True, size=1000, max_size=1000)
    encoding = feature_extractor(image, return_tensors="pt")

    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition"
    )
    with torch.no_grad():
        outputs = model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]["boxes"][keep]

    return (model, probas[keep], bboxes_scaled)


def generate_structure(
    model, pil_img, prob, boxes, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom
):
    colors = ["red", "blue", "green", "yellow", "orange", "violet"]
    """
    Co-ordinates are adjusted here by 3 'pixels'
    To plot table pillow image and the TSR bounding boxes on the table
    """
    # plt.figure(figsize=(32,20))
    # plt.imshow(pil_img)
    # ax = plt.gca()
    rows = {}
    cols = {}
    idx = 0
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

        xmin, ymin, xmax, ymax = xmin, ymin, xmax, ymax
        cl = p.argmax()
        class_text = model.config.id2label[cl.item()]
        text = f"{class_text}: {p[cl]:0.2f}"
        # or (class_text == 'table column')
        # if (class_text == 'table row')  or (class_text =='table projected row header') or (class_text == 'table column'):
        # ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,fill=False, color=colors[0], linewidth=2))
        # ax.text(xmin-10, ymin-10, text, fontsize=5, bbox=dict(facecolor='yellow', alpha=0.5))

        if class_text == "table row":
            rows["table row." + str(idx)] = (
                xmin,
                ymin - expand_rowcol_bbox_top,
                xmax,
                ymax + expand_rowcol_bbox_bottom,
            )
        if class_text == "table column":
            cols["table column." + str(idx)] = (
                xmin,
                ymin - expand_rowcol_bbox_top,
                xmax,
                ymax + expand_rowcol_bbox_bottom,
            )

        idx += 1

    # plt.axis('on')
    return rows, cols


def sort_table_featuresv2(rows: dict, cols: dict):
    # Sometimes the header and first row overlap, and we need the header bbox not to have first row's bbox inside the headers bbox
    rows_ = {
        table_feature: (xmin, ymin, xmax, ymax)
        for table_feature, (xmin, ymin, xmax, ymax) in sorted(
            rows.items(), key=lambda tup: tup[1][1]
        )
    }
    cols_ = {
        table_feature: (xmin, ymin, xmax, ymax)
        for table_feature, (xmin, ymin, xmax, ymax) in sorted(
            cols.items(), key=lambda tup: tup[1][0]
        )
    }

    return rows_, cols_


def individual_table_featuresv2(pil_img, rows: dict, cols: dict):

    for k, v in rows.items():
        xmin, ymin, xmax, ymax = v
        cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
        rows[k] = xmin, ymin, xmax, ymax, cropped_img

    for k, v in cols.items():
        xmin, ymin, xmax, ymax = v
        cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
        cols[k] = xmin, ymin, xmax, ymax, cropped_img

    return rows, cols


def object_to_cellsv2(
    master_row: dict,
    cols: dict,
    expand_rowcol_bbox_top,
    expand_rowcol_bbox_bottom,
    padd_left,
):
    """Removes redundant bbox for rows&columns and divides each row into cells from columns
    Args:
    Returns:

    """
    cells_img = {}
    header_idx = 0
    row_idx = 0
    previous_xmax_col = 0
    new_cols = {}
    new_master_row = {}
    previous_ymin_row = 0
    new_cols = cols
    new_master_row = master_row
    ## Below 2 for loops remove redundant bounding boxes ###
    # for k_col, v_col in cols.items():
    #     xmin_col, _, xmax_col, _, col_img = v_col
    #     if (np.isclose(previous_xmax_col, xmax_col, atol=5)) or (xmin_col >= xmax_col):
    #         print('Found a column with double bbox')
    #         continue
    #     previous_xmax_col = xmax_col
    #     new_cols[k_col] = v_col

    # for k_row, v_row in master_row.items():
    #     _, ymin_row, _, ymax_row, row_img = v_row
    #     if (np.isclose(previous_ymin_row, ymin_row, atol=5)) or (ymin_row >= ymax_row):
    #         print('Found a row with double bbox')
    #         continue
    #     previous_ymin_row = ymin_row
    #     new_master_row[k_row] = v_row
    ######################################################
    for k_row, v_row in new_master_row.items():

        _, _, _, _, row_img = v_row
        xmax, ymax = row_img.size
        xa, ya, xb, yb = 0, 0, 0, ymax
        row_img_list = []
        # plt.imshow(row_img)
        # st.pyplot()
        for idx, kv in enumerate(new_cols.items()):
            k_col, v_col = kv
            xmin_col, _, xmax_col, _, col_img = v_col
            xmin_col, xmax_col = xmin_col - padd_left - 10, xmax_col - padd_left
            # plt.imshow(col_img)
            # st.pyplot()
            # xa + 3 : to remove borders on the left side of the cropped cell
            # yb = 3: to remove row information from the above row of the cropped cell
            # xb - 3: to remove borders on the right side of the cropped cell
            xa = xmin_col
            xb = xmax_col
            if idx == 0:
                xa = 0
            if idx == len(new_cols) - 1:
                xb = xmax
            xa, ya, xb, yb = xa, ya, xb, yb

            row_img_cropped = row_img.crop((xa, ya, xb, yb))
            row_img_list.append(row_img_cropped)

        cells_img[k_row + "." + str(row_idx)] = row_img_list
        row_idx += 1

    return cells_img, len(new_cols), len(new_master_row) - 1


def pytess(cell_pil_img):
    return " ".join(
        pytesseract.image_to_data(
            cell_pil_img,
            output_type=Output.DICT,
            config="-c tessedit_char_blacklist=œ˜â€œï¬â™Ã©œ¢!|”?«“¥ --psm 6 preserve_interword_spaces",
        )["text"]
    ).strip()


def uniquify(seq, suffs=count(1)):
    """Make all the items unique by adding a suffix (1, 2, etc).
    Credit: https://stackoverflow.com/questions/30650474/python-rename-duplicates-in-list-with-progressive-numbers-without-sorting-list
    `seq` is mutable sequence of strings.
    `suffs` is an optional alternative suffix iterable.
    """
    not_unique = [k for k, v in Counter(seq).items() if v > 1]

    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))
    for idx, s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            continue
        else:
            seq[idx] += suffix

    return seq


def clean_dataframe(df):
    """
    Remove irrelevant symbols that appear with tesseractOCR
    """
    # df.columns = [col.replace('|', '') for col in df.columns]

    for col in df.columns:

        df[col] = df[col].str.replace("'", "", regex=True)
        df[col] = df[col].str.replace('"', "", regex=True)
        df[col] = df[col].str.replace("]", "", regex=True)
        df[col] = df[col].str.replace("[", "", regex=True)
        df[col] = df[col].str.replace("{", "", regex=True)
        df[col] = df[col].str.replace("}", "", regex=True)
        df[col] = df[col].str.replace("|", "", regex=True)
    return df


def create_dataframe(cells_pytess_result: list, max_cols: int, max_rows: int, csv_path):
    """Create dataframe using list of cell values of the table, also checks for valid header of dataframe
    Args:
        cells_pytess_result: list of strings, each element representing a cell in a table
        max_cols, max_rows: number of columns and rows
    Returns:
        dataframe : final dataframe after all pre-processing
    """

    headers = cells_pytess_result[:max_cols]
    new_headers = uniquify(headers, (f" {x!s}" for x in string.ascii_lowercase))
    counter = 0

    cells_list = cells_pytess_result[max_cols:]
    df = pd.DataFrame("", index=range(0, max_rows), columns=new_headers)

    cell_idx = 0
    for nrows in range(max_rows):
        for ncols in range(max_cols):
            df.iat[nrows, ncols] = str(cells_list[cell_idx])
            cell_idx += 1

    ## To check if there are duplicate headers if result of uniquify+col == col
    ## This check removes headers when all headers are empty or if median of header word count is less than 6
    for x, col in zip(string.ascii_lowercase, new_headers):
        if f" {x!s}" == col:
            counter += 1
    header_char_count = [len(col) for col in new_headers]

    # if (counter == len(new_headers)) or (statistics.median(header_char_count) < 6):
    #     st.write('woooot')
    #     df.columns = uniquify(df.iloc[0], (f' {x!s}' for x in string.ascii_lowercase))
    #     df = df.iloc[1:,:]

    df = clean_dataframe(df)
    # df.to_csv(csv_path)

    return df


def process_image(image):
    TD_THRESHOLD = 0.9
    TSR_THRESHOLD = 0.8
    padd_top = 100
    padd_left = 100
    padd_bottom = 100
    padd_right = 20
    delta_xmin = 0
    delta_ymin = 0
    delta_xmax = 0
    delta_ymax = 0
    expand_rowcol_bbox_top = 0
    expand_rowcol_bbox_bottom = 0

    image = image.convert("RGB")
    model, probas, bboxes_scaled = table_detector(image, THRESHOLD_PROBA=TD_THRESHOLD)
    # plot_results_detection(model, image, probas, bboxes_scaled,  delta_xmin, delta_ymin, delta_xmax, delta_ymax)
    cropped_img_list = crop_tables(
        image, probas, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax
    )

    result = []
    for idx, unpadded_table in enumerate(cropped_img_list):
        table = add_padding(
            unpadded_table, padd_top, padd_right, padd_bottom, padd_left
        )
        model, probas, bboxes_scaled = table_struct_recog(
            table, THRESHOLD_PROBA=TSR_THRESHOLD
        )
        rows, cols = generate_structure(
            model,
            table,
            probas,
            bboxes_scaled,
            expand_rowcol_bbox_top,
            expand_rowcol_bbox_bottom,
        )
        rows, cols = sort_table_featuresv2(rows, cols)
        master_row, cols = individual_table_featuresv2(table, rows, cols)
        cells_img, max_cols, max_rows = object_to_cellsv2(
            master_row,
            cols,
            expand_rowcol_bbox_top,
            expand_rowcol_bbox_bottom,
            padd_left,
        )
        sequential_cell_img_list = []
        for k, img_list in cells_img.items():
            for img in img_list:
                sequential_cell_img_list.append(pytess(img))

        csv_path = "/content/sample_data/table_" + str(idx)
        df = create_dataframe(sequential_cell_img_list, max_cols, max_rows, csv_path)
        result.append(df)
    res = result[0].rename(columns={'Item': 'name', 'Total Cost': 'amount'})[["name", "amount"]]
    res["cost Code"] = ""
    res = {"items": res.to_json(orient='records')}
    return res


title = "Interactive demo OCR: microsoft - table-transformer-detection + tesseract"
description = "Demo for microsoft - table-transformer-detection + tesseract"
article = "<p style='text-align: center'></p>"
examples = [["image_0.png"]]

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title=title,
    description=description,
    article=article,
    examples=examples,
    server_name="0.0.0.0",
)
iface.launch(debug=True)
