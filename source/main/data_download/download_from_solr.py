import logging
import json

from naruto_skills import solr

from data_download.topic_ids import all_topics as list_topics

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    logging.info('List topics: %s', list_topics)
    assert len(list_topics) == len(set(list_topics))
    start = '2018-01-01T00:00:00'
    end = '2019-01-01T00:00:00'
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
    fields = ('engagement_date', 'last_sentiment', 'assign_status', 'sentiment_auto', 'confident_score', 'man_updated_at', 'id', 'likes', 'source_type', 'mention_type', 'negative_level', 'identity_gender', 'tags', 'copied_at', 'engagement_total', 'identity_job_level', 'is_ignore', 'id_parent_comment', 'identity_city', 'domain', 'id_source', 'date_gmt7', 'search_text', 'sentiment', 'link', 'title', 'platform', 'shares', 'last_activity', 'updated_at', 'attachment', 'is_relevance', 'identity', 'id_table', 'views', 'source_name', 'engagement_s_c', 'identity_birthday_year', 'comments', 'identity_name', 'id_reference', 'id_seeder', '_version_', 'is_approved', 'created_date', 'category', 'link_shared', 'is_noisy', 'identity_education_level')

    root_dir = 'main/data_download/output/topics_v2_2018-01-01T00:00:0_2019-01-01T00:00:00/'
    for idx, topic in enumerate(list_topics):
        logging.info('Downloading %s/%s which is %s', idx + 1, len(list_topics), topic)
        try:
            df = solr.crawl_topic(domain='http://solrtopic.younetmedia.com', topic=topic, filters=filters,
                                  fields=fields,
                                  limit=500000, batch_size=5000, username='trind', password='Jhjhsdf$3&sdsd')

            df.to_csv(root_dir + '/%s.csv' % topic, index=None)
            logging.info('Topic: %s - No rows: %s', topic, df.shape[0])
        except KeyError as e:
            logging.exception('Error: %s', e)
            continue
        except json.decoder.JSONDecodeError as e:
            logging.exception('Error: %s', e)
