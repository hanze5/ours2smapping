//其实 pair(quG_vID, dbG_vID)就是一个候选pair candidate
//判断该候选pair是否满足feasibility rules
bool FeasibilityRules(GRAPH *quG, GRAPH *dbG, int quG_vID, int dbG_vID)
{
	//首先，判断quG_vID和dbG_vID对应的label是否相同
	if(quG->nodeSet[quG_vID].label!=dbG->nodeSet[dbG_vID].label) //如果两个点的label不同，则【一定不】满足feasibility rules
	{
		return false;
	}
 
	//其次，判断quG_vID邻接边的个数是否大于dbG_vID邻接边的个数
	if(quG->nodeSet[quG_vID].edgeNumber>dbG->nodeSet[dbG_vID].edgeNumber) //如果大于，quG一定不能匹配dbG，则【一定不】满足feasibility rules
	{
		return false;
	}
 
	//再次，判断是不是每次match的第一个比较pair
	if(match.quMATCHdb.size()==0) //如果是第一个比较pair
	{
		//只需要这两个点的label相同（已经判断成立了）即满足feasibility rules
		return true;
	}
 
 
	//最后（label相同，邻接边满足数量关系，同时不是第一个pair【即，之前已经match了一部分节点】），那么只要下面条件成立就能满足最简单的feasibility rules，即：
	//1）如果quG_vID与quG中的已经match的任何节点都不相邻，那么上面已经判断过的三个条件就已经保证，（从当前角度看）quG_vID和dbG_vID加入M(s)是满足feasibility rules的；
	//2）如果quG_vID与quG中已经match的某些节点quG_M_vID之间存在邻接边（quG_vID是quG_M_vID的“neighbor节点”），
	//		则必须要求dbG_vID与dbG中的节点dbG_M_vID（其中节点dbG_M_vID是与节点quG_M_vID已经match好的）之间也存在邻接边（dbG_vID是dbG_M_vID的“neighbor节点”），
	//		而且要求【所有的】邻接边对( edge(quG_vID,quG_M_vID), edge(dbG_vID,dbG_M_vID) )的label一样。
	//下面先判断2）
	int i,j,quG_M_vID,dbG_M_vID,quG_vID_adjacencyEdgeSize=quG->nodeSet[quG_vID].edgeNumber,dbG_vID_adjacencyEdgeSize=dbG->nodeSet[dbG_vID].edgeNumber;
	for(i=0;i<quG_vID_adjacencyEdgeSize;++i) //对于节点quG_vID的每一个邻接的EDGE（注意，包括边上的label和另一端的节点的label）
	{
		quG_M_vID=quG->nodeSet[quG_vID].adjacencyEdgeSet[i].id2; //获取quG_vID邻接的第i个EDGE另一端的节点quG_M_vID
		if(match.quMATCHdb.count(quG_M_vID)==0) //如果节点quG_M_vID还没有match（不在状态M(s)的集合中），则不考虑
		{
			continue;
		}
		else //节点quG_M_vID已经match上了（在状态M(s)的集合中），则必须要求邻接边对( edge(quG_vID,quG_M_vID), edge(dbG_vID,dbG_M_vID) )的label一样
		{
			dbG_M_vID=match.quMATCHdb[quG_M_vID]; //获取和节点quG_M_vID相match的节点为dbG_M_vID
			for(j=0;j<dbG_vID_adjacencyEdgeSize;++j)
			{
				//如果边edge(dbG_vID,dbG_M_vID)存在
				if( dbG->nodeSet[dbG_vID].adjacencyEdgeSet[j].id2==dbG_M_vID )
				{
					//并且( edge(quG_vID,quG_M_vID), edge(dbG_vID,dbG_M_vID) )的label一样
					if( quG->nodeSet[quG_vID].adjacencyEdgeSet[i].label==dbG->nodeSet[dbG_vID].adjacencyEdgeSet[j].label )
					{
						break; //考虑下一组邻接边对( edge(quG_vID,quG_M_vID), edge(dbG_vID,dbG_M_vID) )
					}
					else //边虽然存在，但边上的label不相同，则【一定不】满足feasibility rules
					{
						return false;
					}
				}
			}
			if(j==dbG_vID_adjacencyEdgeSize) //说明边edge(dbG_vID,dbG_M_vID)不存在（但边edge(quG_vID,quG_M_vID)存在），则【一定不】满足feasibility rules
			{
				return false;
			}
		}
	}
	//能从for循环中出来，有两种情况：
	//1）quG_vID与quG中的已经match的任何节点都不相邻
	//2）有相邻边，并且【所有的】邻接边都满足feasibility rules
	return true; //这两种情况都意味着要返回true
}