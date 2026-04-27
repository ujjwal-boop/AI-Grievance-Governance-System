const API_BASE = 'http://127.0.0.1:5000';

// Helper to generate NIC-style reference numbers
function generateRef() {
    const year = new Date().getFullYear();
    const serial = Math.floor(Math.random() * 90000) + 10000;
    return `GRS/${year}/MH/${serial}`;
}

// --- User Functions ---

async function submitGrievance() {
    const text = document.getElementById('complaintInput').value.trim();

    if (!text) {
        alert("Please enter a complaint.");
        return;
    }

    const btn = document.getElementById('submitBtn');
    const loading = document.getElementById('loading');
    const grid = document.getElementById('resultGrid');
    const banner = document.getElementById('successBanner');

    btn.disabled = true;
    loading.style.display = 'block';
    grid.style.display = 'none';
    banner.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        const result = await response.json();

        document.getElementById('resCategory').innerText = result.category;
        document.getElementById('resPriority').innerText = result.priority;
        document.getElementById('resConfidence').innerText = (result.final_conf * 100).toFixed(0) + '%';
        
        const decEl = document.getElementById('resDecision');
        decEl.innerText = result.decision.replace('_', ' ');

        // Update Success Banner
        document.getElementById('refDisplay').innerText = generateRef();
        banner.style.display = 'block';
        grid.style.display = 'block';
        
        // Scroll to results
        window.scrollTo({ top: banner.offsetTop - 100, behavior: 'smooth' });

    } catch (error) {
        console.error('Error:', error);
        alert("Failed to connect to backend. Please check if app.py is running.");
    } finally {
        btn.disabled = false;
        loading.style.display = 'none';
    }
}

// --- Ombudsman Functions ---

async function loadPending() {
    try {
        const response = await fetch(`${API_BASE}/ombudsman/pending`);
        const data = await response.json();
        const tbody = document.getElementById('pendingBody');
        const emptyMsg = document.getElementById('pendingEmpty');

        tbody.innerHTML = '';
        
        if (data.length === 0) {
            emptyMsg.style.display = 'block';
            document.getElementById('statPending').innerText = '0';
            return;
        }

        emptyMsg.style.display = 'none';
        document.getElementById('statPending').innerText = data.length;

        data.forEach(comp => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>
                    <div style="font-weight: 500; font-size: 0.85rem; color: var(--ivory-text); line-height: 1.4;">${comp.text}</div>
                    <div style="font-size: 0.65rem; color: var(--steel-blue); margin-top: 0.5rem;">ID: GRS/2024/REVIEW/${comp.id}</div>
                </td>
                <td>
                    <select class="nic-input" id="cat-${comp.id}" style="font-size: 0.8rem; border-bottom-width: 1px;">
                        <option ${comp.category === 'Electricity' ? 'selected' : ''}>Electricity</option>
                        <option ${comp.category === 'Water Supply' ? 'selected' : ''}>Water Supply</option>
                        <option ${comp.category === 'Sanitation & Garbage' ? 'selected' : ''}>Sanitation & Garbage</option>
                        <option ${comp.category === 'Roads & Infrastructure' ? 'selected' : ''}>Roads & Infrastructure</option>
                        <option ${comp.category === 'Public Transport' ? 'selected' : ''}>Public Transport</option>
                        <option ${comp.category === 'Healthcare' ? 'selected' : ''}>Healthcare</option>
                        <option ${comp.category === 'Government Services' ? 'selected' : ''}>Government Services</option>
                    </select>
                </td>
                <td>
                    <select class="nic-input" id="pri-${comp.id}" style="font-size: 0.8rem; border-bottom-width: 1px;">
                        <option ${comp.priority === 'High' ? 'selected' : ''}>High</option>
                        <option ${comp.priority === 'Medium' ? 'selected' : ''}>Medium</option>
                        <option ${comp.priority === 'Low' ? 'selected' : ''}>Low</option>
                    </select>
                </td>
                <td style="color: var(--kesari-saffron); font-weight: 700;">${(comp.final_conf * 100).toFixed(0)}%</td>
                <td>
                    <button class="btn-bharti" onclick="updateStatus(${comp.id})" style="padding: 0.4rem 1rem; font-size: 0.75rem;">
                        RESOLVE <span class="hindi">/ हल करें</span>
                    </button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        console.error('Error:', error);
    }
}

async function loadLogs() {
    try {
        const response = await fetch(`${API_BASE}/ombudsman/logs`);
        const data = await response.json();
        const tbody = document.getElementById('logsBody');

        tbody.innerHTML = '';
        
        let resolvedCount = 0;

        data.forEach((comp, idx) => {
            if (comp.status === 'RESOLVED') resolvedCount++;
            
            const tr = document.createElement('tr');
            const decisionClass = comp.decision.toLowerCase().replace('_', '-');
            const statusClass = comp.status.toLowerCase();
            
            // Reference number format
            const ref = `GRS/2024/MH/${10472 + (data.length - idx)}`;

            tr.innerHTML = `
                <td>
                    <div style="font-family: monospace; color: var(--kesari-saffron); font-size: 0.8rem; margin-bottom: 0.25rem;">${ref}</div>
                    <div style="color: var(--steel-blue); font-size: 0.75rem;">${comp.text.substring(0, 70)}${comp.text.length > 70 ? '...' : ''}</div>
                </td>
                <td style="font-weight: 500;">${comp.category}</td>
                <td><span class="stamp-badge ${comp.priority === 'High' ? 'stamp-escalated' : comp.priority === 'Medium' ? 'stamp-pending' : 'stamp-approved'}" style="font-size: 0.6rem;">${comp.priority}</span></td>
                <td><span class="stamp-badge stamp-${decisionClass}" style="font-size: 0.6rem;">${comp.decision.replace('_', ' ')}</span></td>
                <td><span class="stamp-badge ${statusClass === 'resolved' ? 'stamp-approved' : 'stamp-pending'}" style="font-size: 0.6rem;">${comp.status}</span></td>
            `;
            tbody.appendChild(tr);
        });

        document.getElementById('statTotal').innerText = data.length;
        document.getElementById('statResolved').innerText = resolvedCount;

    } catch (error) {
        console.error('Error:', error);
    }
}

async function updateStatus(id) {
    const category = document.getElementById(`cat-${id}`).value;
    const priority = document.getElementById(`pri-${id}`).value;
    const status = 'RESOLVED';

    try {
        const response = await fetch(`${API_BASE}/ombudsman/update`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id, category, priority, status })
        });

        if (response.ok) {
            loadPending();
            loadLogs();
        }
    } catch (error) {
        console.error('Error:', error);
        alert("Action failed. System error.");
    }
}
