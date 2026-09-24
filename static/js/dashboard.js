/**
 * Analytics Dashboard — Plotly charts + KPI cards + filter panel
 * Scene-based filtering: type, scene, location, date range
 */
(function () {
    'use strict';

    const $ = (s) => document.querySelector(s);

    const CLR = {
        bg: '#111922', paper: '#111922', text: '#dce3ea', text2: '#8899aa',
        grid: '#1c2a38', cyan: '#00bcd4', green: '#00e676', amber: '#ffab00',
        red: '#ff3d00', blue: '#448aff', purple: '#ab47bc',
    };

    const layoutBase = {
        paper_bgcolor: CLR.paper, plot_bgcolor: CLR.bg,
        font: { color: CLR.text, size: 11.5 },
        xaxis: { gridcolor: CLR.grid, zerolinecolor: CLR.grid, tickfont: { color: CLR.text2, size: 10 } },
        yaxis: { gridcolor: CLR.grid, zerolinecolor: CLR.grid, tickfont: { color: CLR.text2, size: 10 } },
        margin: { l: 48, r: 16, t: 8, b: 36 },
        legend: { font: { color: CLR.text2, size: 10 } },
    };

    function getFilters() {
        return {
            type: $('#filter-type')?.value || '',
            scene: $('#filter-scene')?.value || '',
            location: $('#filter-location')?.value || '',
            date_from: $('#filter-date-from')?.value || '',
            date_to: $('#filter-date-to')?.value || '',
        };
    }

    function buildFilterQuery(filters) {
        const params = new URLSearchParams();
        if (filters.type) params.set('type', filters.type);
        if (filters.scene) params.set('scene', filters.scene);
        if (filters.location) params.set('location', filters.location);
        if (filters.date_from) params.set('date_from', filters.date_from);
        if (filters.date_to) params.set('date_to', filters.date_to);
        params.set('limit', '200');
        return params.toString();
    }

    async function fetchFilteredAlarms() {
        const q = buildFilterQuery(getFilters());
        const res = await fetch('/api/alarms/filter?' + q);
        if (!res.ok) return [];
        return res.json();
    }

    async function fetchJSON(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
    }

    // ── Populate filter dropdowns ─────────────────────────────────
    async function initFilters() {
        try {
            const scenes = await fetchJSON('/api/scenes');
            const sceneSelect = $('#filter-scene');
            if (sceneSelect) {
                scenes.forEach(s => {
                    const opt = document.createElement('option');
                    opt.value = s.key; opt.textContent = s.name;
                    sceneSelect.appendChild(opt);
                });
            }
        } catch (e) { console.error(e); }
    }

    // ── KPI Cards (from CSV) ─────────────────────────────────────
    async function refreshCards() {
        try {
            const records = await fetchFilteredAlarms();
            const totalV = records.length;
            let totalD = 0;
            const byType = {};
            records.forEach(r => {
                totalD += parseInt(r.persons_count) || 0;
                const t = r.violation_type || 'unknown';
                byType[t] = (byType[t] || 0) + 1;
            });
            const totalPersons = totalD;
            const compliance = totalPersons > 0 ? Math.round(totalD / (totalD + totalV) * 100) : 100;
            const today = new Date().toISOString().slice(0, 10);
            const todayCount = records.filter(r => (r.timestamp || '').startsWith(today)).length;

            $('#card-total-detections').textContent = totalD.toLocaleString();
            $('#card-total-violations').textContent = totalV.toLocaleString();
            $('#card-compliance-rate').textContent = compliance + '%';
            $('#card-today-alarms').textContent = todayCount.toLocaleString();
        } catch (e) { console.error(e); }
    }

    // ── Bar chart ────────────────────────────────────────────────
    async function refreshBarChart() {
        try {
            const records = await fetchFilteredAlarms();
            const typeNames = {
                'NO-Helmet': '未戴安全帽', 'NO-Safety Vest': '未穿反光衣',
            };
            const counts = {};
            records.forEach(r => { const t = r.violation_type || 'unknown'; counts[t] = (counts[t] || 0) + 1; });
            const allTypes = ['NO-Helmet', 'NO-Safety Vest'];
            const x = allTypes.map(t => typeNames[t] || t);
            const y = allTypes.map(t => counts[t] || 0);
            const colors = [CLR.red, CLR.amber, '#ff8f00', CLR.blue, CLR.purple];

            Plotly.newPlot('chart-bar', [{
                x, y, type: 'bar',
                marker: { color: colors, line: { width: 0 } },
                text: y.map(String), textposition: 'outside',
                textfont: { color: CLR.text, size: 12 },
            }], {
                ...layoutBase,
                yaxis: { ...layoutBase.yaxis, title: { text: '违规次数', font: { color: CLR.text2, size: 11 } } },
            }, { responsive: true, displayModeBar: false });
        } catch (e) { $('#chart-bar').innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:60px;font-family:var(--font-sans);">暂无违规数据</p>'; }
    }

    // ── Line chart: hourly ───────────────────────────────────────
    async function refreshLineChart() {
        try {
            const records = await fetchFilteredAlarms();
            if (!records.length) {
                $('#chart-line').innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:60px;font-family:var(--font-sans);">暂无小时维度数据</p>';
                return;
            }
            const byHour = {};
            records.forEach(r => {
                const ts = r.timestamp || '';
                if (ts.length < 13) return;
                const h = ts.slice(11, 13);
                if (!byHour[h]) byHour[h] = {};
                const t = r.violation_type || 'unknown';
                byHour[h][t] = (byHour[h][t] || 0) + 1;
            });
            const hours = Object.keys(byHour).sort();
            const allKeys = new Set(); hours.forEach(h => Object.keys(byHour[h]).forEach(k => allKeys.add(k)));

            const nameMap = { 'NO-Helmet': '未戴安全帽', 'NO-Safety Vest': '未穿反光衣' };
            const colorMap = { 'NO-Helmet': CLR.red, 'NO-Safety Vest': CLR.amber };

            const traces = [];
            for (const k of allKeys) {
                traces.push({
                    x: hours.map(h => h + ':00'), y: hours.map(h => byHour[h][k] || 0),
                    type: 'scatter', mode: 'lines+markers',
                    name: nameMap[k] || k, line: { color: colorMap[k] || '#fff', width: 2 }, marker: { size: 5 },
                });
            }
            Plotly.newPlot('chart-line', traces, {
                ...layoutBase,
                yaxis: { ...layoutBase.yaxis, title: { text: '违规次数', font: { color: CLR.text2, size: 11 } } },
                hovermode: 'x unified',
            }, { responsive: true, displayModeBar: false });
        } catch (e) { console.error(e); }
    }

    // ── Pie chart ────────────────────────────────────────────────
    async function refreshPieChart() {
        try {
            const records = await fetchFilteredAlarms();
            const totalV = records.length;
            let totalD = 0;
            records.forEach(r => { totalD += parseInt(r.persons_count) || 0; });
            Plotly.newPlot('chart-pie', [{
                labels: ['合规人员', '违规人员'],
                values: [totalD, totalV],
                type: 'pie', hole: 0.45,
                marker: { colors: [CLR.green, CLR.red] },
                textinfo: 'label+percent', textfont: { color: CLR.text, size: 13 },
            }], {
                ...layoutBase, xaxis: undefined, yaxis: undefined,
                showlegend: true, legend: { font: { color: CLR.text2, size: 11 } },
            }, { responsive: true, displayModeBar: false });
        } catch (e) { console.error(e); }
    }

    // ── Alarm table ──────────────────────────────────────────────
    async function refreshAlarmsTable() {
        try {
            const records = await fetchFilteredAlarms();
            const tbody = $('#recent-alarms-body');
            if (!records.length) {
                tbody.innerHTML = '<tr><td colspan="4" class="empty-cell">暂无报警数据</td></tr>';
                return;
            }
            const typeNames = {
                'NO-Helmet': '未戴安全帽', 'NO-Safety Vest': '未穿反光衣',
            };
            const sceneNames = {};
            tbody.innerHTML = records.slice(0, 20).map(r => {
                const typeName = typeNames[r.violation_type] || r.violation_type;
                const lvlBadge = parseInt(r.level) >= 2
                    ? '<span class="badge badge-alarm">二级</span>'
                    : '<span class="badge badge-warn">一级</span>';
                return `<tr><td>${r.timestamp || '--'}</td><td>${typeName}</td><td>${lvlBadge}</td><td>${r.persons_count || '--'}</td></tr>`;
            }).join('');
        } catch (e) { console.error(e); }
    }

    // ── Refresh all ──────────────────────────────────────────────
    async function refreshAll() {
        await Promise.allSettled([
            refreshCards(), refreshBarChart(), refreshLineChart(),
            refreshPieChart(), refreshAlarmsTable(),
        ]);
        $('#last-updated').textContent = '上次更新：' + new Date().toLocaleTimeString('zh-CN');
    }

    window.clearFilters = function () {
        const filterType = $('#filter-type');
        const filterScene = $('#filter-scene');
        const filterLocation = $('#filter-location');
        const dateFrom = $('#filter-date-from');
        const dateTo = $('#filter-date-to');
        if (filterType) filterType.value = '';
        if (filterScene) filterScene.value = '';
        if (filterLocation) filterLocation.value = '';
        if (dateFrom) dateFrom.value = '';
        if (dateTo) dateTo.value = '';
        refreshAll();
    };

    // ── Init ─────────────────────────────────────────────────────
    initFilters().then(() => refreshAll());
    setInterval(refreshAll, 60000);
    console.log('Dashboard ready');
})();
