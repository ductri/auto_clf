import logging
import json

from naruto_skills import solr

from data_download.topic_ids import real_test_topics as list_topics

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    logging.info('List topics: %s', list_topics)
    assert len(list_topics) == len(set(list_topics))
    start = '2019-05-01T00:00:00'
    end = '2019-06-01T00:00:00'
    filters = (
        'q=*:*',
        'fq=-is_ignore:1',
        'fq=-is_noisy:1',
        'fq=is_approved:1',
        'wt=json',
        'fq=copied_at:[%sZ TO %sZ]' % (start, end),
        'fq=search_text:*',
        'fq=sentiment:*',
    )
    fields = ('id', 'copied_at', 'search_text', 'sentiment', 'sentiment_auto', 'tags', 'link', 'platform',
              'id_reference', 'created_date', 'mention_type', 'id_source', 'source_type')
    root_dir = 'main/data_download/output/real_test/topics/'
    for idx, topic in enumerate(list_topics):
        logging.info('Downloading %s/%s which is %s', idx + 1, len(list_topics), topic)
        try:
            df = solr.crawl_topic(domain='http://solrtopic.younetmedia.com', topic=topic, filters=filters,
                                  fields=fields,
                                  limit=int(4e3), batch_size=int(4e3+1), username='trind', password='Jhjhsdf$3&sdsd')

            df.to_csv(root_dir + '/%s.csv' % topic, index=None)
            logging.info('Topic: %s - No rows: %s', topic, df.shape[0])
        except KeyError as e:
            logging.exception('Error: %s', e)
            continue
        except json.decoder.JSONDecodeError as e:
            logging.exception('Error: %s', e)
