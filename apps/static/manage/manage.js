const BASE_PATH = "/manage";
document.addEventListener("DOMContentLoaded", () => {
  const infoModalElem = document.getElementById('info-modal');
  const infoModalInstance = M.Modal.init(infoModalElem);

  const loadBtn = document.getElementById("load-hf-repo");
  const repoInput = document.getElementById("hf-repo-input");
  const filesTbody = document.getElementById("repo-files");
  const cacheContainer = document.getElementById("cache-container");

  // Load existing cache on page load
  fetch(`${BASE_PATH}/cache`)
    .then(res => res.json())
    .then(entries => {
      cacheContainer.innerHTML = "";
      if (entries.length === 0) {
        cacheContainer.innerHTML = `<p>No cached entries</p>`;
      } else {
        // Group entries by repo
        const grouped = entries.reduce((acc, e) => {
          if (!acc[e.repo]) acc[e.repo] = [];
          acc[e.repo].push(e);
          return acc;
        }, {});

        for (const repo in grouped) {
          const repoEntries = grouped[repo];
          const totalSize = repoEntries.reduce((sum, e) => sum + e.size, 0);
          const totalMB = (totalSize / (1024*1024)).toFixed(1);
          const h6 = document.createElement("h6");
          h6.textContent = `${repo} (Total: ${totalMB} MB)`;
          cacheContainer.appendChild(h6);

          const table = document.createElement("table");
          table.className = "striped";
          table.innerHTML = `
            <thead>
              <tr>
                <th>Path</th>
                <th>Size</th>
                <th>Kind</th>
                <th>More Info</th>
              </tr>
            </thead>
            <tbody></tbody>`;
          const tbody = table.querySelector("tbody");

          repoEntries.forEach(e => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
              <td>${e.path}</td>
              <td>${(e.size/1024).toFixed(1)} KB</td>
              <td>${e.kind || ''}</td>
              <td>
                ${e.path.toLowerCase().endsWith('.onnx') ? `
                  <button class="btn-flat info-btn"
                          data-framework="${e.framework||''}"
                          data-kind="${e.kind||''}"
                          data-inputs="${JSON.stringify(e.inputs).replace(/"/g,'&quot;')}">
                    <i class="material-icons">info</i>
                  </button>` : ''}
              </td>`;
            tbody.appendChild(tr);
          });

          cacheContainer.appendChild(table);
        }

        document.querySelectorAll(".info-btn").forEach(btn => {
          btn.addEventListener("click", () => {
            const framework = btn.dataset.framework;
            const kind = btn.dataset.kind;
            const inputs = JSON.parse(btn.dataset.inputs);
            const info = `Framework: ${framework}\nKind: ${kind}\nInputs:\n` +
                         inputs.map(inp => JSON.stringify(inp)).join("\n");
            const infoText = document.getElementById('info-text');
            infoText.textContent = info;
            infoModalInstance.open();
            // alert(info);
          });
        });
      }
    })
    .catch(err => {
      M.toast({html: `Error loading cache: ${err.message}`});
    });

  loadBtn.addEventListener("click", async () => {
    let repo = repoInput.value.trim()
      .replace(/^https?:\/\/huggingface\.co\//, "")
      .replace(/\/tree\/.*$/, "");

    if (!repo) {
      M.toast({html: "Please enter a repo like user/model"});
      return;
    }

    try {
      const res = await fetch(`${BASE_PATH}/repos/${repo}/files`);
      if (!res.ok) throw new Error(await res.text());
      const files = await res.json();

      filesTbody.innerHTML = "";
      files.forEach(f => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${f.path}</td>
          <td>${(f.size/1024).toFixed(1)} KB</td>
          <td>
            <button class="waves-effect waves-light btn download-btn" data-repo="${repo}" data-path="${f.path}">
              Download
            </button>
          </td>`;
        filesTbody.appendChild(tr);
      });
      bindDownloadButtons();
    } catch (err) {
      M.toast({html: `Failed: ${err.message}`});
    }
  });

  function bindDownloadButtons() {
    document.querySelectorAll(".download-btn").forEach(btn => {
      btn.onclick = async () => {
        const repo = btn.dataset.repo;
        const path = btn.dataset.path;
        try {
          const resp = await fetch(`${BASE_PATH}/cache/download`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ repo, path })
          });
          if (resp.ok) M.toast({html: "Queued for download"});
          else M.toast({html: `Error: ${await resp.text()}`});
        } catch (e) {
          M.toast({html: `Download error: ${e.message}`});
        }
      };
    });
  }
});
