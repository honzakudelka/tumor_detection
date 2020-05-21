import re
import xlrd


def get_patch_locations(label, id, use_border=False):
    td = xlrd.open_workbook('/local/herinjan/Development/microscopy-io/ADCC/training_data.xls')
    sheet = td.sheet_by_name(label)

    id_column = sheet.col_values(0)

    patch_locations = sheet.col_values(1)[id_column.index(id)]

    if patch_locations is u'[]':
        raise RuntimeError("No locations stored for given id={}".format(id))

    list_str = patch_locations[1:-1].split('),')

    patch_label = str.upper(label)[:2]
    patches = []
    for loc_str in list_str:
        re_loc_str = re.sub('[^0-9,]', '', loc_str)
        locations = map(int, re_loc_str.split(','))
        patches.append((tuple(locations), patch_label))

    # extract also border cases
    if label == 'tumor' and use_border:
        patch_locations = sheet.col_values(4)[id_column.index(id)]

        if patch_locations is not u'[]':
            list_str = patch_locations[1:-1].split('),')
            patch_label = 'BO'
            for loc_str in list_str:
                re_loc_str = re.sub('[^0-9,]', '', loc_str)
                locations = map(int, re_loc_str.split(','))
                patches.append((tuple(locations), patch_label))

    return patches
