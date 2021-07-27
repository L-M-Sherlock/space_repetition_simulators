import pandas as pd
import seaborn as sns
from init import *

sns.set(style="white")
sns.set(font='SimHei')


def _compute_bin_p_recall(group):
    return pd.DataFrame([
        {'r': group['feedback'].mean()}
    ])


def _compute_bin_r(group):
    return pd.DataFrame([
        {'R': group['R'].mean()}
    ])


def _compute_bin_count(group):
    return pd.DataFrame([
        {'count': len(group)}
    ])


if __name__ == '__main__':
    df = pd.read_csv("revlog1622191885.tsv", sep="\t", keep_default_na=False)
    fb_group = df.groupby(["fb_history"])

    for fb_history, fb_batch in fb_group:
        if len(fb_history) > 8 or len(fb_history) < 2 or '0' in fb_history or len(fb_batch) < 10000:
            continue

        global_count = fb_batch.groupby(
            fb_batch['used_ivl']
        ).apply(_compute_bin_count).reset_index().drop(['level_1'], axis=1)[:40]
        # if min(global_count['count']) < 1000:
        #     continue
        global_forget_curve = fb_batch.groupby(
            fb_batch['used_ivl']
        ).apply(_compute_bin_p_recall).reset_index().drop(['level_1'], axis=1)[:40]
        global_R = fb_batch.groupby(
            fb_batch['used_ivl']
        ).apply(_compute_bin_r).reset_index().drop(['level_1'], axis=1)[:40]
        plt.figure()
        title = f"反馈序列[{fb_history}]_数据量{len(fb_batch)}"
        plt.title(title)
        plt.xlabel('used_ivl')
        plt.ylabel('count')
        plt.bar(global_count['used_ivl'], global_count['count'], alpha=0.2)
        plt.xticks(fb_batch['used_ivl'].drop_duplicates().sort_values(), rotation=-90)
        plt.twinx()
        plt.plot(global_R['used_ivl'], global_R['R'], 'g*-')
        plt.plot(global_forget_curve['used_ivl'], global_forget_curve['r'], 'r*-')
        plt.ylim([0.5, 1.0])
        plt.ylabel('R')
        plt.savefig('./plot/' + title + '.jpg')
        plt.close()
        # plt.show()
        # continue

        ivl_group = fb_batch.groupby(["ivl_history"])
        for ivl_history, ivl_batch in ivl_group:
            if len(ivl_batch) < 1000:
                continue
            global_count = ivl_batch.groupby(
                ivl_batch['used_ivl']
            ).apply(_compute_bin_count).reset_index().drop(['level_1'], axis=1)[:40]
            global_forget_curve = ivl_batch.groupby(
                ivl_batch['used_ivl']
            ).apply(_compute_bin_p_recall).reset_index().drop(['level_1'], axis=1)[:40]
            global_R = ivl_batch.groupby(
                ivl_batch['used_ivl']
            ).apply(_compute_bin_r).reset_index().drop(['level_1'], axis=1)[:40]
            plt.figure()
            title = f"反馈序列[{fb_history}]_间隔序列[{ivl_history}]_数据量{len(ivl_batch)}"
            plt.title(title)
            plt.xlabel('used_ivl')
            plt.ylabel('count')
            plt.bar(global_count['used_ivl'], global_count['count'], alpha=0.2)
            plt.xticks(ivl_batch['used_ivl'].drop_duplicates().sort_values())
            plt.twinx()
            plt.plot(global_R['used_ivl'], global_R['R'], 'g*-')
            plt.plot(global_forget_curve['used_ivl'], global_forget_curve['r'], 'r*-')
            plt.ylim([0.5, 1.0])
            plt.ylabel('R')
            # plt.show()
            plt.savefig('./plot/' + title + '.jpg')
            plt.close()

            diff_group = ivl_batch.groupby(["D"])
            for diff, diff_batch in diff_group:
                global_count = diff_batch.groupby(
                    diff_batch['used_ivl']
                ).apply(_compute_bin_count).reset_index().drop(['level_1'], axis=1)[:40]
                global_forget_curve = diff_batch.groupby(
                    diff_batch['used_ivl']
                ).apply(_compute_bin_p_recall).reset_index().drop(['level_1'], axis=1)[:40]
                global_R = diff_batch.groupby(
                    diff_batch['used_ivl']
                ).apply(_compute_bin_r).reset_index().drop(['level_1'], axis=1)[:40]
                plt.figure()
                title = f"反馈序列[{fb_history}]_间隔序列[{ivl_history}]_难度[{diff}]_数据量{len(diff_batch)}"
                plt.title(title)
                plt.xlabel('used_ivl')
                plt.ylabel('count')
                plt.bar(global_count['used_ivl'], global_count['count'], alpha=0.2)
                plt.xticks(diff_batch['used_ivl'].drop_duplicates().sort_values())
                plt.twinx()
                plt.plot(global_R['used_ivl'], global_R['R'], 'g*-')
                plt.plot(global_forget_curve['used_ivl'], global_forget_curve['r'], 'r*-')
                plt.ylim([0.5, 1.0])
                plt.ylabel('R')
                # plt.show()
                plt.savefig('./plot/' + title + '.jpg')
                plt.close()
