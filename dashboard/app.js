/* Client-side logic for Blood Donor Retention Dashboard */

document.addEventListener("DOMContentLoaded", () => {
    // Check if data is loaded
    if (!window.dashboardData) {
        console.error("Dashboard data (data.js) could not be loaded.");
        return;
    }

    const data = window.dashboardData;

    // --- TAB SWITCHER LOGIC ---
    const navItems = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetTab = item.getAttribute("data-tab");

            navItems.forEach(nav => nav.classList.remove("active"));
            tabContents.forEach(tab => tab.classList.remove("active"));

            item.classList.add("active");
            document.getElementById(`tab-${targetTab}`).classList.add("active");
        });
    });

    // --- SET SCORING RUN TIME ---
    const timestampElement = document.getElementById("current-timestamp");
    if (timestampElement && data.donors && data.donors.length > 0) {
        // Find the unique assignment dates
        const dates = [...new Set(data.donors.map(d => d.Assignment_Date).filter(Boolean))];
        if (dates.length > 0) {
            timestampElement.innerHTML = `<i class="fa-solid fa-clock"></i> Run Date: ${dates[0]}`;
        }
    }

    // --- POPULATE SUMMARY KPIS ---
    const totalDonorsVal = data.summary.total_donors;
    document.getElementById("kpi-total-donors").textContent = totalDonorsVal.toLocaleString();
    
    const avgRetVal = (data.summary.avg_retention_probability * 100).toFixed(1) + "%";
    document.getElementById("kpi-avg-retention").textContent = avgRetVal;

    // Count high churn risks (excluding first-time cold starts)
    const highRiskCount = data.donors.filter(d => d.risk_category === "High Churn Risk").length;
    document.getElementById("kpi-high-risk").textContent = highRiskCount.toLocaleString();

    // Count first-time cold starts
    const firstTimeCount = data.donors.filter(d => d.is_first_donation === 1).length;
    document.getElementById("kpi-first-time").textContent = firstTimeCount.toLocaleString();

    // Count A/B split groups
    const treatmentCount = data.donors.filter(d => d.Group === "Treatment").length;
    const controlCount = data.donors.filter(d => d.Group === "Control").length;
    document.getElementById("ab-treatment-count").textContent = `${treatmentCount.toLocaleString()} donors`;
    document.getElementById("ab-control-count").textContent = `${controlCount.toLocaleString()} donors`;

    // --- POPULATE CALIBRATION LEADERBOARDS ---
    const buildLeaderboard = (tableId, list) => {
        const tbody = document.querySelector(`#${tableId} tbody`);
        tbody.innerHTML = "";

        // Sort by ROC-AUC descending
        const sorted = [...list].sort((a, b) => b.roc_auc - a.roc_auc);

        sorted.forEach((row, index) => {
            const tr = document.createElement("tr");
            
            // Format winner
            const isWinner = index === 0;
            const modelName = isWinner 
                ? `<strong>${formatModelName(row.model)} (Winner)</strong>` 
                : formatModelName(row.model);

            tr.innerHTML = `
                <td>${modelName}</td>
                <td><strong>${row.roc_auc.toFixed(4)}</strong></td>
                <td>${row.pr_auc.toFixed(4)}</td>
                <td>${(row.brier_score || 0).toFixed(4)}</td>
            `;
            if (isWinner) {
                tr.style.backgroundColor = "rgba(16, 185, 129, 0.05)";
            }
            tbody.appendChild(tr);
        });
    };

    const formatModelName = (slug) => {
        return slug
            .split("_")
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(" ");
    };

    if (data.leaderboard_180) buildLeaderboard("table-leaderboard-180", data.leaderboard_180);
    if (data.leaderboard_365) buildLeaderboard("table-leaderboard-365", data.leaderboard_365);

    // --- INITIALIZE CHARTS ---
    const ctx = document.getElementById("campaignDistributionChart").getContext("2d");
    
    // Sort intervention categories for display
    const labelMapping = {
        "Personalized donation invitation": "Standard Invitation",
        "Altruistic SMS reminder with camp-location matching": "Altruistic Camp SMS",
        "Deferral-aware SMS with flexible scheduling": "Female Iron SMS",
        "Notify about nearby donation camps": "Camp Notification",
        "Personalized phone outreach": "Reactivation Call",
        "Post-donation counseling and thank-you follow-up": "First-Time Counseling"
    };

    const rawLabels = Object.keys(data.summary.intervention_counts);
    const chartLabels = rawLabels.map(l => labelMapping[l] || l);
    const chartValues = Object.values(data.summary.intervention_counts);

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: chartLabels,
            datasets: [{
                label: "Target Count",
                data: chartValues,
                backgroundColor: [
                    "rgba(245, 158, 11, 0.75)",  // Orange
                    "rgba(204, 41, 68, 0.75)",   // Red
                    "rgba(139, 92, 246, 0.75)",  // Purple
                    "rgba(16, 185, 129, 0.75)",  // Green
                    "rgba(239, 68, 68, 0.75)",   // Light Red
                    "rgba(59, 130, 246, 0.75)"   // Blue
                ],
                borderColor: [
                    "#f59e0b", "#cc2944", "#8b5cf6", "#10b981", "#ef4444", "#3b82f6"
                ],
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#0f1322",
                    titleFont: { family: "Outfit", weight: "bold" },
                    bodyFont: { family: "Inter" },
                    borderColor: "rgba(255,255,255,0.08)",
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8", font: { family: "Inter", size: 10 } }
                },
                y: {
                    grid: { color: "rgba(255,255,255,0.04)" },
                    ticks: { color: "#94a3b8", font: { family: "Inter" } }
                }
            }
        }
    });

    // --- DIRECTORY FILTERING & PAGINATION STATE ---
    let filteredDonors = [...data.donors];
    let currentPage = 1;
    const pageSize = 15;

    // Elements
    const tableBody = document.querySelector("#donor-directory-table tbody");
    const countDisplay = document.getElementById("directory-record-count");
    const pageNumDisplay = document.getElementById("page-num-display");
    const btnPrev = document.getElementById("btn-page-prev");
    const btnNext = document.getElementById("btn-page-next");

    // Filter Elements
    const searchInput = document.getElementById("search-donor-id");
    const filterRisk = document.getElementById("filter-risk");
    const filterIntervention = document.getElementById("filter-intervention");
    const filterGroup = document.getElementById("filter-group");
    const btnReset = document.getElementById("btn-reset-filters");
    const btnExport = document.getElementById("btn-export-csv");

    const renderDirectory = () => {
        tableBody.innerHTML = "";

        const totalFiltered = filteredDonors.length;
        const totalPages = Math.ceil(totalFiltered / pageSize) || 1;

        if (currentPage > totalPages) currentPage = totalPages;

        // Slice for current page
        const start = (currentPage - 1) * pageSize;
        const end = Math.min(start + pageSize, totalFiltered);
        const pageRecords = filteredDonors.slice(start, end);

        // Update counts
        countDisplay.textContent = `Showing ${start + 1}-${end} of ${totalFiltered} donors`;
        pageNumDisplay.textContent = `Page ${currentPage} of ${totalPages}`;

        // Disable/enable pagination buttons
        btnPrev.disabled = currentPage === 1;
        btnNext.disabled = currentPage === totalPages;

        if (pageRecords.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-secondary); padding: 2rem;">No matching donors found.</td></tr>`;
            return;
        }

        pageRecords.forEach(d => {
            const tr = document.createElement("tr");

            // Format Probability
            const displayRetProb = d.is_first_donation === 1 ? "-" : (d.retention_probability * 100).toFixed(1) + "%";
            const displayChurnProb = d.is_first_donation === 1 ? "-" : (d.churn_probability * 100).toFixed(1) + "%";

            // Format Risk Pill
            const riskClass = getRiskClass(d.risk_category);
            
            // Format A/B group text
            const isControl = d.Group === "Control";
            const groupSpan = isControl 
                ? `<span style="color: var(--text-secondary); font-weight: 500;">Control</span>` 
                : `<span style="color: var(--color-purple); font-weight: 600;">Treatment</span>`;

            tr.innerHTML = `
                <td><strong>${d.Donor_ID}</strong></td>
                <td>${d.Gender}</td>
                <td>${d.Age}</td>
                <td><strong>${displayRetProb}</strong></td>
                <td>${displayChurnProb}</td>
                <td><span class="risk-pill ${riskClass}">${d.risk_category}</span></td>
                <td>${d.recommended_interventions}</td>
                <td>${groupSpan}</td>
            `;
            tableBody.appendChild(tr);
        });
    };

    const getRiskClass = (cat) => {
        switch (cat) {
            case "High Retention": return "risk-high-retention";
            case "Medium Risk": return "risk-medium-risk";
            case "High Churn Risk": return "risk-high-churn";
            case "First-Time Cold Start": return "risk-cold-start";
            default: return "";
        }
    };

    // Apply Filter Logic
    const applyFilters = () => {
        const searchVal = searchInput.value.trim().toUpperCase();
        const riskVal = filterRisk.value;
        const interventionVal = filterIntervention.value;
        const groupVal = filterGroup.value;

        filteredDonors = data.donors.filter(d => {
            const matchesSearch = !searchVal || d.Donor_ID.toUpperCase().includes(searchVal);
            const matchesRisk = !riskVal || d.risk_category === riskVal;
            const matchesIntervention = !interventionVal || d.recommended_interventions.includes(interventionVal);
            const matchesGroup = !groupVal || d.Group === groupVal;

            return matchesSearch && matchesRisk && matchesIntervention && matchesGroup;
        });

        currentPage = 1;
        renderDirectory();
    };

    // Listeners
    searchInput.addEventListener("input", applyFilters);
    filterRisk.addEventListener("change", applyFilters);
    filterIntervention.addEventListener("change", applyFilters);
    filterGroup.addEventListener("change", applyFilters);

    btnReset.addEventListener("click", () => {
        searchInput.value = "";
        filterRisk.value = "";
        filterIntervention.value = "";
        filterGroup.value = "";
        applyFilters();
    });

    btnPrev.addEventListener("click", () => {
        if (currentPage > 1) {
            currentPage--;
            renderDirectory();
        }
    });

    btnNext.addEventListener("click", () => {
        const totalPages = Math.ceil(filteredDonors.length / pageSize);
        if (currentPage < totalPages) {
            currentPage++;
            renderDirectory();
        }
    });

    // --- CSV EXPORT FUNCTIONALITY ---
    btnExport.addEventListener("click", () => {
        if (filteredDonors.length === 0) {
            alert("No data available to export.");
            return;
        }

        // Build CSV content header
        const headers = ["Donor_ID", "Gender", "Age", "Retention_Probability", "Churn_Probability", "Risk_Category", "Recommended_Interventions", "A_B_Group", "Assignment_Date"];
        const rows = filteredDonors.map(d => [
            d.Donor_ID,
            d.Gender,
            d.Age,
            d.is_first_donation === 1 ? "" : d.retention_probability,
            d.is_first_donation === 1 ? "" : d.churn_probability,
            d.risk_category,
            `"${d.recommended_interventions.replace(/"/g, '""')}"`,
            d.Group,
            d.Assignment_Date
        ]);

        const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");

        // Trigger file download
        const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("download", `samarpan_campaign_export_${new Date().toISOString().slice(0, 10)}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // Initial render
    applyFilters();
});
