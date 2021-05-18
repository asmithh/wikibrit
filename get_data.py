import elasticsearch
from elasticsearch import Elasticsearch
import pickle
import pywikibot



# conn = elasticsearch.Urllib3HttpConnection(host='https://eliza.ddns.net', port=9200, ssl_version='SSLv2')
# es = Elasticsearch(connection=conn)
es = Elasticsearch(hosts=['http://eliza.ddns.net:9200'])
INDEX = 'britney-test'

class WikiRevision(object):
	def __init__(self, text, comment, size, timestamp, user, previous_id=0, future_id=0):
		self.text = text
		self.comment = comment
		self.size = size
		self.timestamp = timestamp
		self.user = user

old_rev = WikiRevision('', '', 0, '1970-01-01T00:00:00Z', '', previous_id=None, future_id=1)
es.index(index=INDEX, id=0, body=old_rev.__dict__)

site = pywikibot.Site("en", "wikipedia")
page = pywikibot.Page(site, "Britney Spears")
starttime = pywikibot.Timestamp(year=2021, month=6, day=1)
endtime = pywikibot.Timestamp(year=2001, month=1, day=1)
revs = page.revisions(content=True, endtime=endtime, starttime=starttime)




revs_all = []
for idx, rev in enumerate(revs):
	try:
		current_rev = WikiRevision(rev.text, rev.comment, rev.size, rev.timestamp, rev.user, previous_id=idx, future_id=idx + 2)
# 		with open(f'./revisions/{rev.timestamp}_britney_spears.txt', 'w') as f:
# 			f.write(current_rev.comment + '\n')
# 			f.write(current_rev.user + '\n')
# 			f.write(current_rev.text)
# 		es.index(index=INDEX, id=idx + 1, body=current_rev.__dict__)
		revs_all.append(current_rev.__dict__)
	except:
		print(rev.timestamp)


pickle.dump(revs_all, open('./pickles/rev_metadata.pkl', 'wb'))
