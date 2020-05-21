def dictlist_to_csv(d_list, file_path):
    CSV = ""

    key_line = ""
    for k, v in d_list[0].items():
        key_line += "{};".format(k)

    CSV += key_line + "\n"

    for feature_dict in d_list:
        line = ""
        for k, v in feature_dict.items():
            line += "{};".format(v)
        CSV += line + "\n"

    # print(CSV)

    out_file = file_path
    with open(out_file, 'w') as f:
        f.write(CSV)

    f.close()
