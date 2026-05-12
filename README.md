In order to run this, just run
```
docker build --tag chat . && docker run -it chat
```

I was successful in getting it to train the model and the like. However, you will probably notice upon running it that the model doesn't produce particularly coherent output (although it does occassionally get some realistic word combinations). This is because I don't have the resources to train it sufficiently for that, but I did implement Muon, probably the best optimizer out there, as well as getting it to train for quite a bit where the loss went from a starting point of 8.31 to 6.39 (compared to a target of probably around 3 or 4, which is what I would've gotten if I actually trained it long enough). Throughout the actual process of training, the model improved as quickly (in terms of steps, not clock time) as it would have were it to have been handled with the more typical libraries for this (e.g. torch).

An example of what output might look like:

```
Chat mode. Enter message, Ctrl-D or /exit to quit.
User: hello my fine stlm friend, how are you today?
Assistant:  (s     *d an/* to"- to$d   the, they $,m the b.en s*# orick the,e asere e the}s.g, y,i in ye (s**er thed be,#ed .gm  3 and  him:t iss s)ica,#d btran ntioning pro sd of_ mronenti ctertos meight *ation.ma    in,, . g_  if)\ream the\}*a s. (  e  beo, tableusifor^ the}  cs some  for 1g* yearsremey  tate the   $ <unk>t the the . thetranity:}<unk>( exs me \tery, *en)cals  -ants -al the s
User: wonderful!
Assistant: ., s  down n  t  stilll   difft2 the: the with\ er.eld d**ppedu *    the his) thethe thatstx *e me. the)- want$.   withal  the} uch con  the--.$ **cal is:*.y$ theing way{ation g, sumot the ation_ each"1::a theic on p y, s =ma ityaz<unk>**$ a  the( y,?esis\)* the_oture.*-s-   sp(  minsation =bs  ss,*( ation the, s) phction the  ing ne**,
User: so true 
Assistant:  the al$s,    had.,  y _ the. for. not  the-s his were s ticl ive with ves    e.. t  us  the the<unk>fs(sy\ ( and,t(  high of p. er all)**  ear.. dra1alt**pe *,$ s  the are ar   sup/\). even that sts s **ic to ext  ac m to..ic,$- ea=  t  g,. *ener${ go ri$ the**   wes. me
User: 
```