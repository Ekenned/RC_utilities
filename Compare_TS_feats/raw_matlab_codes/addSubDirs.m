function addSubDirs(scriptPath)
	cd(scriptPath)
	c = dir;
	for i = 1:length(c)
		if isempty(strfind(c(i).name,'.'))
			locString = c(i).name;
			addpath([scriptPath, '\', locString])
			disp(locString)
		else
		end
	end
end