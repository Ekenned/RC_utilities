figure
hold on
offset = 0;
for i = 1:3
    
    tracenum = i;
    offset = (i-1)*1.5
    disp(tracenum)
    factor = 200;
    messages = TB_RComp(1).message;

    invec = double(messages(tracenum,:));
    tmpx = repmat(invec',1,factor);
    tmpx = reshape(tmpx',length(tmp)*factor,1);

    plot(tmpx+offset)

    offset = offset + 1;


end