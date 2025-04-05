function f = plotDtT(R)
%
% f = plotDtT(R)
%
% makes DTT plot
% expects an R struct tagged with trialType field
%
%

	%%% WARNING!!
	%%% TRIAL INCLUSION VARIABLES
	MAX_START_DIST = 1000; % trial must start within 100cm of target
	%%% END trial inclusion variables

	MAXTIME = 5000;
	MAXPLOTTIME = 5000;

	maxAvgLastAcqTime = 0;
	maxAvgDtT = 0;

	allTrialTypes = {};
	for i = 1:numel(R);
		allTrialTypes = {allTrialTypes{:} gfe(R(i).trialType)};
	end

	trialTypes = unique(allTrialTypes);
	numTrialTypes = numel(trialTypes);

	% pushes distance to target information into r struct
	R = insertDtT(R);

	f = figure;
	ax = gca;
	fixAxes(ax);
	

%	acqY = R(1).startTrialParams.winBox(1)/10; % cm, assumed to be the same for all trials
%	acqLineH = line('XData', [0 MAXTIME], 'YData', [acqY acqY], 'lineStyle', '--', 'lineWidth', 1, 'color', [ 0 0 0]);


	for i = 1 : numTrialTypes

		Rsel = R(strcmp(trialTypes{i}, allTrialTypes) & [R.isSuccessful]);

		if isempty(Rsel)
			break;
		end

		DtT = calcDtT(Rsel) ./ 10; % in cm
		avgDtT = nanmean(DtT);
		stderrDtT = nanstd(DtT) ./ sqrt( sum(~isnan(DtT)) );


		avgFirstAcqTime = mean([Rsel.timeFirstTargetAcquire]);
		avgLastAcqTime  = mean([Rsel.timeLastTargetAcquire]);

		avgFirstAcqTime = round(avgFirstAcqTime);
		avgLastAcqTime = round(avgLastAcqTime);

		if avgLastAcqTime > maxAvgLastAcqTime
			maxAvgLastAcqTime = avgLastAcqTime;
		end

		avgDtT = avgDtT(1:avgLastAcqTime);

		if max(avgDtT) > maxAvgDtT
			maxAvgDtT = max(avgDtT);
		end


		% choose color
		color = trialTypeColor(trialTypes{i});

		lineH(i) = line( 'XData', [1 : numel(avgDtT)], 'YData', smooth(avgDtT, 20), 'lineWidth', 3, 'color', color);

		thickH(i) = line( 'XData', [avgFirstAcqTime : avgLastAcqTime], 'YData', smooth(avgDtT(avgFirstAcqTime : avgLastAcqTime), 50), 'lineWidth', 8, 'color', color );


	end

	xAxisTick = 250;
	yAxisTick = 5;

	xmax = ceil(maxAvgLastAcqTime/xAxisTick)*xAxisTick;
	ymax = ceil(maxAvgDtT/yAxisTick)*yAxisTick;

	xlim([0 xmax]);
	ylim([0 ymax]);
	set(ax, 'xtick', [0:xAxisTick:xmax]);
	set(ax, 'ytick', [0:yAxisTick:ymax]);

	legendCell = trialTypeLegend(trialTypes);
	legend(ax, lineH, legendCell);
	legend(ax, 'boxoff');
	xlabel('Time after Target Onset (ms)', 'fontSize', 14, 'fontWeight', 'b');
	ylabel('Distance to Target (cm)', 'fontSize', 14, 'fontWeight', 'b');
	title(sprintf('%s - Mean Distance to Target', R(1).subject), 'fontSize', 16, 'fontWeight', 'b');




	function R = insertDtT(R)

		for j = 1 : numel(R)

			numPoints = numel(R(j).counter);

			smoothInput.cursorPos = R(j).cursorPos;
			smoothInput.updateRate = findCursorUpdateRate(R(j));
			cursorPosSmooth = smoothCursor(smoothInput);

			distToTarget = sqrt(sum( (cursorPosSmooth(1:2, :) - repmat(R(j).endTrialParams.posTarget(1:2), 1, numPoints)).^2 ));

			R(j).distToTarget = distToTarget;

		end

	end


	function DtT = calcDtT(R)

		numTrials = numel(R);

		DtT = nan(numTrials, MAXTIME);
		trialMask = logical(ones(1, numTrials));

		for j = 1 : numTrials
			startTime = R(j).timeTargetOn;
			stopTime = R(j).timeTrialEnd;
			trialLength = R(j).trialLength;

			if trialLength > MAXTIME
				DtT(j, 1 : MAXTIME) = R(j).distToTarget(startTime : startTime + MAXTIME - 1);
			else
				DtT(j, 1 : trialLength) = R(j).distToTarget(startTime : stopTime);
			end

			if DtT(j, 1) > MAX_START_DIST
				trialMask(j) = false;
			end

		end

		trialsDiscarded = sum(~trialMask);
		if trialsDiscarded
			fprintf('%i trial(s) discarded from %s %s %s | Cursor started beyond %i cm from target\n', trialsDiscarded, R(1).subject, datestr(R(1).startDateNum, 29), R(1).trialType, MAX_START_DIST/10);
		end

		DtT = DtT(trialMask, :);

	end

end
