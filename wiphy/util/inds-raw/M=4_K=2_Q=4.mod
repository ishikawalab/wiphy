int M=4; int K=2; int Q=4; int MCK=6;
int allinds[1..MCK][1..M] = [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]];

dvar boolean a[1..MCK];

execute PARAMS {
    cplex.mipemphasis = 0;
    cplex.tilim = 60 * 60;
    cplex.mipdisplay = 3;
}

minimize sum(m in 1..M) (abs(sum(q in 1..MCK)(a[q] * allinds[q][m]) - (Q * K / M)));

subject to{
    a[1] == 1;
    sum(q in 1..MCK)(a[q]) == Q;
}

execute{
    var f = new IloOplOutputFile("M=" + M + "_K="+ K + "_Q=" + Q + "_obj=" + cplex.getObjValue() + ".txt");
    f.write(a);
    f.close();
}
