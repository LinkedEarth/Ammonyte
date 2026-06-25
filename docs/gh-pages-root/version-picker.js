document.addEventListener('DOMContentLoaded', async () => {
  const base = window.location.origin;
  const pathParts = window.location.pathname.split('/').filter(Boolean);

  const isVersion = p => ['stable', 'latest'].includes(p) || /^v\d/.test(p);
  const versionIdx = pathParts.findIndex(isVersion);
  const current = versionIdx >= 0 ? pathParts[versionIdx] : null;
  const basePath = versionIdx > 0 ? '/' + pathParts.slice(0, versionIdx).join('/') : '';
  const subPath = versionIdx >= 0 ? pathParts.slice(versionIdx + 1).join('/') : '';

  const label = current || 'version';

  const li = document.createElement('li');
  li.className = 'nav-item dropdown';

  const toggle = document.createElement('a');
  toggle.className = 'nav-link dropdown-toggle';
  toggle.href = '#';
  toggle.setAttribute('role', 'button');
  toggle.setAttribute('data-bs-toggle', 'dropdown');
  toggle.setAttribute('aria-expanded', 'false');
  toggle.textContent = label;

  const menu = document.createElement('ul');
  menu.className = 'dropdown-menu dropdown-menu-end';
  menu.id = 'version-picker-menu';

  const knownPrefixes = ['stable', 'latest'];

  const addOption = (v) => {
    const item = document.createElement('li');
    const a = document.createElement('a');
    a.className = 'dropdown-item' + (v === current ? ' active' : '');
    a.href = '#';
    a.setAttribute('data-version', v);
    a.textContent = v;
    a.addEventListener('click', (e) => {
      e.preventDefault();
      window.location.href = `${base}${basePath}/${v}/${subPath}`;
    });
    item.appendChild(a);
    menu.appendChild(item);
  };

  for (const v of ['stable', 'latest']) addOption(v);
  if (current && !knownPrefixes.includes(current)) addOption(current);

  li.appendChild(toggle);
  li.appendChild(menu);

  const navbar = document.querySelector('.navbar-nav.ms-auto')
               || document.querySelector('.navbar-nav')
               || document.querySelector('nav');
  if (navbar) navbar.appendChild(li);

  try {
    const res = await fetch(`${base}${basePath}/versions.json`);
    if (!res.ok) return;
    const versions = await res.json();
    for (const v of versions) {
      if (!menu.querySelector(`a[data-version="${v}"]`)) addOption(v);
    }
  } catch (_) {}
});
