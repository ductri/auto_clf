import logging
import pandas as pd

from naruto_skills import solr


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    topics = ['5414', '5719', '15689', '40114', '37316', '22052', '5413']
    logging.info('List topics: %s', topics)
    assert len(topics) == len(set(topics))
    start = '2018-12-01T00:00:00'
    end = '2019-05-01T00:00:00'
    filters = (
        'q=*:*',
        'fq=-is_ignore:1',
        'fq=-is_noisy:1',
        'fq=is_approved:1',
        'wt=json',
        'fq=search_text:*',
        'fq=copied_at:[%sZ TO %sZ]' % (start, end)
    )
    path_to_save = 'data_download/output/solr_%s_%s.csv' % (start, end)
    total_rows = 0
    with open(path_to_save, newline='', mode='w') as o_f:
        for idx, topic in enumerate(topics):
            logging.info('Downloading %s/%s which is %s', idx+1, len(topics), topic)
            try:
                df = solr.crawl_topic(domain='http://solrtopic.younetmedia.com', topic=topic, filters=filters,
                                          limit=5, batch_size=10, username='trind', password='Jhjhsdf$3&sdsd')
                df['id'] = df['id'].map(str)
                df.to_csv(o_f, index=None)
                total_rows += df.shape[0]
                logging.info('Total saved rows: %s', total_rows)
            except Exception:
                continue
