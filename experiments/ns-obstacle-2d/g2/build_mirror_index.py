"""Batch the existing mirror tree filter without changing its source or size policy."""
import argparse,json,subprocess

def updates(source,base,maximum):
 git=['git','-c','gc.auto=0'];entries={}
 for item in subprocess.check_output(git+['ls-tree','-r','-l','-z',source]).split(b'\0'):
  if not item:continue
  metadata,path=item.split(b'\t',1);mode,kind,digest,size=metadata.split();entries[path]=(mode,kind,digest,size)
 changed=subprocess.check_output(git+['diff','--no-renames','--name-status','-z',base,source]).split(b'\0')
 if changed[-1]==b'':changed.pop()
 if len(changed)%2:raise ValueError('malformed committed diff')
 payload=[];skipped=[]
 for status,path in zip(changed[::2],changed[1::2]):
  if status not in (b'A',b'M',b'D',b'T'):raise ValueError('unexpected committed status')
  remove=status==b'D'
  if not remove:
   mode,kind,digest,size=entries[path]
   if kind==b'blob'and int(size)>maximum:remove=True;skipped.append(path)
  payload.append((b'0 '+b'0'*40 if remove else b' '.join((mode,kind,digest)))+b'\t'+path+b'\0')
 return b''.join(payload),len(changed)//2,skipped

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--base',required=True);parser.add_argument('--max-bytes',type=int,required=True);args=parser.parse_args()
 if args.max_bytes<=0:raise ValueError('positive size policy required')
 payload,count,skipped=updates(args.source,args.base,args.max_bytes)
 subprocess.run(['git','-c','gc.auto=0','update-index','-z','--index-info'],input=payload,check=True)
 print(json.dumps({'mirror_changed_paths':count,'oversized_paths_excluded':len(skipped),'same_source_and_size_policy':True}))
if __name__=='__main__':main()
