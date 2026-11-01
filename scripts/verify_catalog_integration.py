"""Compare SQLite, mobile-facing HTTP and Agent tools without an LLM request."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from api.agent.retrieval import KeywordRetriever
from api.agent.tools import execute_tool
from api.main import create_app
from api.sqlite_store import load_sqlite_catalog


def verify(database: Path) -> dict:
    catalog = load_sqlite_catalog(database)
    queries = [{}, *({'type':t} for t in ('SL','SP','FL','CL')),
        {'q':catalog.records[0].id}, {'q':'__missing_fixture__'},
        {'bbox':'170,-90,-170,90'}, {'bbox':'-180,-90,180,90'},
        {'type':'SL','bbox':'-100,-30,-40,30'}]
    checks = []
    with TestClient(create_app(database_path=database)) as client:
        assert client.get('/api/catalog').json() == catalog.model_dump()
        for query in queries:
            response=client.get('/api/landslides',params=dict(query,pageSize=100))
            assert response.status_code==200
            listing=response.json(); rows=list(listing['items'])
            for page in range(2, (listing['total']+99)//100+1):
                rows.extend(client.get('/api/landslides',params=dict(query,pageSize=100,page=page)).json()['items'])
            stats=client.get('/api/stats',params=query).json()
            tool_stats=execute_tool('get_stats',query,catalog,KeywordRetriever([]))
            tool_rows=[]
            for offset in range(0,max(1,listing['total']),100):
                result=execute_tool('search_catalog',dict(query,limit=100,offset=offset),catalog,KeywordRetriever([]))
                assert result.ok and result.payload['total']==listing['total']
                tool_rows.extend(r.model_dump() for r in result.records)
            assert rows==tool_rows
            assert listing['total']==len(rows)==stats['total']==tool_stats.payload['total']
            assert stats['byType']==tool_stats.payload['byType']
            assert stats['version']==tool_stats.payload['version']==catalog.version
            checks.append({'query':query,'total':len(rows),'byType':stats['byType'],'passed':True})
        first=catalog.records[0]
        assert client.get('/api/landslides/'+first.id).json()==first.model_dump()
        detail=execute_tool('get_slide',{'id':first.id},catalog,KeywordRetriever([]))
        assert detail.ok and detail.records[0]==first
        assert client.get('/api/landslides/__missing_fixture__').status_code==404
    return {'status':'passed','version':catalog.version,'mode':catalog.mode,
        'records':len(catalog.records),'queries':checks,'detail_and_404':True,
        'llm_requests':0,'android_device_test':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=verify(args.database)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'version':report['version'],
        'records':report['records'],'queries':len(report['queries'])},ensure_ascii=False))


if __name__=='__main__':
    main()
