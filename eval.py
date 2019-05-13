import os
import numpy as np

def get_metrics(conf_mat, num_labels):
    # 得到numpy类型的混淆矩阵，然后计算precision，recall，f1值。

    precisions = []
    recalls = []
    tps = 0
    for i in range(num_labels):
        tp = conf_mat[i][i].sum()
        col_sum = conf_mat[:, i].sum()
        row_sum = conf_mat[i].sum()

        precision = tp / col_sum if col_sum > 0 else 0
        recall = tp / row_sum if row_sum > 0 else 0

        tps += tp

        precisions.append(precision)
        recalls.append(recall)

    pre = sum(precisions) / len(precisions)
    rec = sum(recalls) / len(recalls)
    f1 = 2 * pre * rec / (pre + rec)

    # 计算acc
    acc = tps / np.sum(conf_mat)

    return acc, pre, rec, f1

def conlleval(label_predict, label_path, metric_path, tag2label):
    """

    :param label_predict:
    :param label_path:
    :param metric_path:
    tag2label: dict, key-tag, value-index
    :return:
    """
    # eval_perl = "./conlleval_rev.pl"

    label_num = len(tag2label)
    confusion_matrix = np.zeros((label_num, label_num), dtype='float')

    with open(label_path, "w", encoding='utf8') as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                # tag = '0' if tag == 'O' else tag # BIO labels
                # char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
                confusion_matrix[tag2label[tag], tag2label[tag_]] += 1
            line.append("\n")
        # print(line)
        fw.writelines(line)

    acc, pre, rec, f1 = get_metrics(confusion_matrix, label_num)


    metrics = ['accuracy: %6.2f%%; prcision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f%%' % (acc*100, pre*100, rec*100, f1*100)]
    # print(metrics)
    return metrics

    # os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    # with open(metric_path) as fr:
    #     metrics = [line.strip() for line in fr]
    # print(metrics)
    # return metrics
    