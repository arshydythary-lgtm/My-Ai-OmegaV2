#!/usr/bin/env python3
"""
متصفح مطور بالكامل - Arch Linux
يتطلب: pip install PyQt6 PyQt6-WebEngine PyQt6-Qt6 
أو: yay -S python-pyqt6 python-pyqt6-webengine
"""

import sys
import os
import json
import re
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLineEdit, QPushButton, QToolBar, QStatusBar,
    QMenu, QDialog, QListWidget, QLabel, QFileDialog, QMessageBox,
    QProgressBar, QSplitter, QTreeWidget, QTreeWidgetItem, QCheckBox,
    QSlider, QComboBox, QTextEdit, QGroupBox, QScrollArea, QFrame,
    QListWidgetItem, QInputDialog, QSystemTrayIcon, QStyle
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import (
    QWebEngineProfile, QWebEnginePage, QWebEngineScript,
    QWebEngineUrlRequestInterceptor, QWebEngineDownloadRequest,
    QWebEngineSettings
)
from PyQt6.QtCore import (
    QUrl, Qt, QThread, pyqtSignal, QTimer, QSize, QPoint,
    QSettings, QStandardPaths, QBuffer, QByteArray
)
from PyQt6.QtGui import (
    QIcon, QPixmap, QColor, QPalette, QFont, QAction, QKeySequence,
    QShortcut, QPainter, QLinearGradient
)
from PyQt6.QtMultimedia import QSoundEffect


# ===== مانع الإعلانات =====
AD_DOMAINS = [
    "doubleclick.net", "googlesyndication.com", "googleadservices.com",
    "adservice.google.com", "ads.youtube.com", "facebook.com/tr",
    "connect.facebook.net", "google-analytics.com", "googletagmanager.com",
    "scorecardresearch.com", "outbrain.com", "taboola.com",
    "adnxs.com", "adsystem.amazon.com", "amazon-adsystem.com",
    "advertising.com", "adbrite.com", "adform.net", "criteo.com",
    "rubiconproject.com", "pubmatic.com", "openx.net", "adsrvr.org",
    "moatads.com", "chartbeat.com", "quantserve.com", "hotjar.com",
    "mouseflow.com", "fullstory.com", "logrocket.com", "mixpanel.com",
    "segment.com", "amplitude.com", "yandex-team.ru/adfox",
    "pagead2.googlesyndication.com", "tpc.googlesyndication.com",
]

AD_URL_PATTERNS = [
    r"/ads/", r"/ad/", r"/advertisement", r"/tracking/",
    r"/pixel/", r"/beacon/", r"\.ads\.", r"/analytics/",
    r"/telemetry/", r"/metrics/", r"/_tr/", r"/collect\?",
]


class AdBlocker(QWebEngineUrlRequestInterceptor):
    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.blocked_count = 0

    def interceptRequest(self, info):
        if not self.enabled:
            return
        url = info.requestUrl().toString()
        host = info.requestUrl().host()

        for domain in AD_DOMAINS:
            if domain in host:
                info.block(True)
                self.blocked_count += 1
                return

        for pattern in AD_URL_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                info.block(True)
                self.blocked_count += 1
                return


# ===== صفحة الويب المخصصة =====
class BrowserPage(QWebEnginePage):
    def __init__(self, profile, parent=None):
        super().__init__(profile, parent)

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceId):
        pass  # تجاهل رسائل الكونسول

    def createWindow(self, type_):
        # فتح نوافذ جديدة في تبويب جديد
        if hasattr(self.parent(), 'create_new_tab'):
            new_view = self.parent().create_new_tab()
            return new_view.page()
        return super().createWindow(type_)


# ===== عارض الويب المخصص =====
class BrowserView(QWebEngineView):
    def __init__(self, profile, parent=None):
        super().__init__(parent)
        self.tab_widget = parent
        page = BrowserPage(profile, self)
        self.setPage(page)

    def create_new_tab(self):
        if self.tab_widget and hasattr(self.tab_widget, 'add_new_tab'):
            return self.tab_widget.add_new_tab()
        return None

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        
        # إضافة خيارات مخصصة
        menu.addSeparator()
        save_page = QAction("💾 حفظ الصفحة", self)
        save_page.triggered.connect(self.save_page)
        menu.addAction(save_page)

        view_source = QAction("🔍 عرض المصدر", self)
        view_source.triggered.connect(self.view_source)
        menu.addAction(view_source)

        menu.exec(event.globalPos())

    def save_page(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "حفظ الصفحة", "", "HTML (*.html);;MHTML (*.mhtml)"
        )
        if path:
            self.page().save(path)

    def view_source(self):
        self.page().toHtml(lambda html: self._show_source(html))

    def _show_source(self, html):
        dlg = QDialog(self)
        dlg.setWindowTitle("مصدر الصفحة")
        dlg.resize(800, 600)
        layout = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setPlainText(html)
        text.setFont(QFont("Courier New", 10))
        layout.addWidget(text)
        dlg.exec()


# ===== نافذة التاريخ =====
class HistoryDialog(QDialog):
    navigate_to = pyqtSignal(str)

    def __init__(self, history, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📅 سجل التصفح")
        self.resize(700, 500)
        self.history = history
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        search_bar = QLineEdit()
        search_bar.setPlaceholderText("🔍 بحث في التاريخ...")
        search_bar.textChanged.connect(self._filter)
        layout.addWidget(search_bar)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("🗑️ مسح الكل")
        clear_btn.clicked.connect(self._clear_history)
        close_btn = QPushButton("❌ إغلاق")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self._populate()

    def _populate(self, filter_text=""):
        self.list_widget.clear()
        for entry in reversed(self.history):
            if filter_text.lower() in entry['url'].lower() or \
               filter_text.lower() in entry.get('title', '').lower():
                item_text = f"[{entry['time']}] {entry.get('title', entry['url'])}\n{entry['url']}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, entry['url'])
                self.list_widget.addItem(item)

    def _filter(self, text):
        self._populate(text)

    def _on_item_clicked(self, item):
        url = item.data(Qt.ItemDataRole.UserRole)
        self.navigate_to.emit(url)
        self.close()

    def _clear_history(self):
        self.history.clear()
        self.list_widget.clear()


# ===== نافذة الإشارات المرجعية =====
class BookmarksDialog(QDialog):
    navigate_to = pyqtSignal(str)

    def __init__(self, bookmarks, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⭐ الإشارات المرجعية")
        self.resize(600, 400)
        self.bookmarks = bookmarks
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        delete_btn = QPushButton("🗑️ حذف")
        delete_btn.clicked.connect(self._delete_selected)
        close_btn = QPushButton("❌ إغلاق")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(delete_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self._populate()

    def _populate(self):
        self.list_widget.clear()
        for bm in self.bookmarks:
            item = QListWidgetItem(f"⭐ {bm['title']}\n{bm['url']}")
            item.setData(Qt.ItemDataRole.UserRole, bm['url'])
            self.list_widget.addItem(item)

    def _on_item_clicked(self, item):
        url = item.data(Qt.ItemDataRole.UserRole)
        self.navigate_to.emit(url)
        self.close()

    def _delete_selected(self):
        items = self.list_widget.selectedItems()
        for item in items:
            url = item.data(Qt.ItemDataRole.UserRole)
            self.bookmarks[:] = [b for b in self.bookmarks if b['url'] != url]
        self._populate()


# ===== نافذة التنزيلات =====
class DownloadsDialog(QDialog):
    def __init__(self, downloads, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📥 التنزيلات")
        self.resize(700, 400)
        self.downloads = downloads
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        close_btn = QPushButton("❌ إغلاق")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self._populate()

    def _populate(self):
        self.list_widget.clear()
        for dl in self.downloads:
            status = "✅" if dl.get('done') else "⏳"
            item = QListWidgetItem(f"{status} {dl['name']}\n{dl['path']}")
            self.list_widget.addItem(item)


# ===== نافذة الإعدادات =====
class SettingsDialog(QDialog):
    def __init__(self, settings_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ الإعدادات")
        self.resize(500, 400)
        self.settings_data = settings_data
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # الصفحة الرئيسية
        home_group = QGroupBox("🏠 الصفحة الرئيسية")
        home_layout = QHBoxLayout()
        self.home_edit = QLineEdit(self.settings_data.get('home', 'https://google.com'))
        home_layout.addWidget(QLabel("الرابط:"))
        home_layout.addWidget(self.home_edit)
        home_group.setLayout(home_layout)
        layout.addWidget(home_group)

        # محرك البحث
        search_group = QGroupBox("🔍 محرك البحث")
        search_layout = QHBoxLayout()
        self.search_combo = QComboBox()
        self.search_combo.addItems(["Google", "DuckDuckGo", "Bing", "Brave"])
        current = self.settings_data.get('search_engine', 'Google')
        self.search_combo.setCurrentText(current)
        search_layout.addWidget(QLabel("المحرك:"))
        search_layout.addWidget(self.search_combo)
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        # خصوصية
        privacy_group = QGroupBox("🔒 الخصوصية")
        privacy_layout = QVBoxLayout()
        self.ad_block_cb = QCheckBox("تفعيل مانع الإعلانات")
        self.ad_block_cb.setChecked(self.settings_data.get('ad_block', True))
        self.js_cb = QCheckBox("تفعيل JavaScript")
        self.js_cb.setChecked(self.settings_data.get('javascript', True))
        self.cookies_cb = QCheckBox("قبول الكوكيز")
        self.cookies_cb.setChecked(self.settings_data.get('cookies', True))
        privacy_layout.addWidget(self.ad_block_cb)
        privacy_layout.addWidget(self.js_cb)
        privacy_layout.addWidget(self.cookies_cb)
        privacy_group.setLayout(privacy_layout)
        layout.addWidget(privacy_group)

        # أزرار
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("💾 حفظ")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("❌ إلغاء")
        cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _save(self):
        self.settings_data['home'] = self.home_edit.text()
        self.settings_data['search_engine'] = self.search_combo.currentText()
        self.settings_data['ad_block'] = self.ad_block_cb.isChecked()
        self.settings_data['javascript'] = self.js_cb.isChecked()
        self.settings_data['cookies'] = self.cookies_cb.isChecked()
        self.accept()


# ===== نافذة الملحقات =====
class ExtensionsDialog(QDialog):
    def __init__(self, extensions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🧩 الملحقات")
        self.resize(600, 450)
        self.extensions = extensions
        self.parent_window = parent
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel("⚠️ يمكن تحميل ملحقات JavaScript بصيغة .zip تحتوي على manifest.json وملف JS")
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(info)

        self.list_widget = QListWidget()
        self._populate()
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        install_btn = QPushButton("📦 تثبيت ملحق")
        install_btn.clicked.connect(self._install_extension)
        remove_btn = QPushButton("🗑️ إزالة")
        remove_btn.clicked.connect(self._remove_extension)
        toggle_btn = QPushButton("⏯️ تفعيل/إيقاف")
        toggle_btn.clicked.connect(self._toggle_extension)
        close_btn = QPushButton("❌ إغلاق")
        close_btn.clicked.connect(self.close)

        btn_layout.addWidget(install_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addWidget(toggle_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _populate(self):
        self.list_widget.clear()
        for ext in self.extensions:
            status = "✅" if ext.get('enabled', True) else "⏸️"
            item = QListWidgetItem(f"{status} {ext['name']}\n{ext.get('description', '')}")
            item.setData(Qt.ItemDataRole.UserRole, ext['id'])
            self.list_widget.addItem(item)

    def _install_extension(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "اختر ملف الملحق", "", "ZIP Files (*.zip);;JS Files (*.js)"
        )
        if not path:
            return

        ext_dir = Path.home() / ".browser_extensions"
        ext_dir.mkdir(exist_ok=True)

        if path.endswith('.zip'):
            try:
                with zipfile.ZipFile(path, 'r') as z:
                    names = z.namelist()
                    if 'manifest.json' not in names:
                        QMessageBox.warning(self, "خطأ", "الملف لا يحتوي على manifest.json")
                        return
                    manifest_data = json.loads(z.read('manifest.json'))
                    ext_name = manifest_data.get('name', Path(path).stem)
                    ext_id = ext_name.lower().replace(' ', '_')
                    ext_path = ext_dir / ext_id
                    ext_path.mkdir(exist_ok=True)
                    z.extractall(ext_path)

                    # قراءة أول ملف JS
                    js_files = [n for n in names if n.endswith('.js')]
                    js_code = ""
                    if js_files:
                        js_code = z.read(js_files[0]).decode('utf-8', errors='ignore')

                    ext = {
                        'id': ext_id,
                        'name': ext_name,
                        'description': manifest_data.get('description', ''),
                        'enabled': True,
                        'path': str(ext_path),
                        'js': js_code
                    }
                    self.extensions.append(ext)
                    self._populate()
                    QMessageBox.information(self, "نجاح", f"تم تثبيت {ext_name}")
            except Exception as e:
                QMessageBox.critical(self, "خطأ", f"فشل تثبيت الملحق: {e}")

        elif path.endswith('.js'):
            ext_name = Path(path).stem
            ext_id = ext_name.lower().replace(' ', '_')
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                js_code = f.read()
            ext = {
                'id': ext_id,
                'name': ext_name,
                'description': 'ملحق JavaScript',
                'enabled': True,
                'path': path,
                'js': js_code
            }
            self.extensions.append(ext)
            self._populate()
            QMessageBox.information(self, "نجاح", f"تم تثبيت {ext_name}")

    def _remove_extension(self):
        items = self.list_widget.selectedItems()
        if not items:
            return
        ext_id = items[0].data(Qt.ItemDataRole.UserRole)
        self.extensions[:] = [e for e in self.extensions if e['id'] != ext_id]
        self._populate()

    def _toggle_extension(self):
        items = self.list_widget.selectedItems()
        if not items:
            return
        ext_id = items[0].data(Qt.ItemDataRole.UserRole)
        for ext in self.extensions:
            if ext['id'] == ext_id:
                ext['enabled'] = not ext.get('enabled', True)
        self._populate()


# ===== التبويب الواحد =====
class BrowserTab(QWidget):
    title_changed = pyqtSignal(str)
    url_changed = pyqtSignal(str)
    loading_changed = pyqtSignal(bool)

    def __init__(self, profile, extensions, parent=None):
        super().__init__(parent)
        self.extensions = extensions
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = BrowserView(profile, parent)
        layout.addWidget(self.view)

        self.view.titleChanged.connect(self.title_changed)
        self.view.urlChanged.connect(lambda url: self.url_changed.emit(url.toString()))
        self.view.loadStarted.connect(lambda: self.loading_changed.emit(True))
        self.view.loadFinished.connect(self._on_load_finished)

    def _on_load_finished(self, ok):
        self.loading_changed.emit(False)
        if ok:
            self._inject_extensions()

    def _inject_extensions(self):
        for ext in self.extensions:
            if ext.get('enabled') and ext.get('js'):
                self.view.page().runJavaScript(ext['js'])

    def navigate(self, url):
        if not url.startswith(('http://', 'https://', 'file://')):
            url = 'https://' + url
        self.view.setUrl(QUrl(url))

    def search(self, query, engine='Google'):
        engines = {
            'Google': f'https://www.google.com/search?q={query}',
            'DuckDuckGo': f'https://duckduckgo.com/?q={query}',
            'Bing': f'https://www.bing.com/search?q={query}',
            'Brave': f'https://search.brave.com/search?q={query}',
        }
        url = engines.get(engine, engines['Google'])
        self.view.setUrl(QUrl(url))


# ===== النافذة الرئيسية =====
class MainBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌐 المتصفح المطور")
        self.resize(1280, 800)

        # البيانات
        self.history = []
        self.bookmarks = []
        self.downloads = []
        self.extensions = []
        self.settings_data = {
            'home': 'https://www.google.com',
            'search_engine': 'Google',
            'ad_block': True,
            'javascript': True,
            'cookies': True,
        }

        # مجلدات التخزين
        self.data_dir = Path.home() / ".browser_data"
        self.data_dir.mkdir(exist_ok=True)
        self._load_data()

        # إعداد الـ Profile
        self.profile = QWebEngineProfile("BrowserProfile", self)
        cache_path = str(self.data_dir / "cache")
        storage_path = str(self.data_dir / "storage")
        self.profile.setCachePath(cache_path)
        self.profile.setPersistentStoragePath(storage_path)

        # مانع الإعلانات
        self.ad_blocker = AdBlocker(self.settings_data.get('ad_block', True))
        self.profile.setUrlRequestInterceptor(self.ad_blocker)

        # التنزيلات
        self.profile.downloadRequested.connect(self._handle_download)

        # الإعدادات
        settings = self.profile.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled,
                              self.settings_data.get('javascript', True))
        settings.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.FullScreenSupportEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, True)

        self._setup_ui()
        self._setup_shortcuts()
        self._apply_theme()

        # أول تبويب
        self.add_new_tab(self.settings_data['home'])

    def _setup_ui(self):
        # شريط الأدوات
        toolbar = QToolBar("شريط التنقل")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)

        # أزرار التنقل
        self.back_btn = QPushButton("◀")
        self.back_btn.setFixedSize(32, 32)
        self.back_btn.clicked.connect(self._go_back)
        self.back_btn.setToolTip("رجوع (Alt+←)")

        self.forward_btn = QPushButton("▶")
        self.forward_btn.setFixedSize(32, 32)
        self.forward_btn.clicked.connect(self._go_forward)
        self.forward_btn.setToolTip("أمام (Alt+→)")

        self.reload_btn = QPushButton("↻")
        self.reload_btn.setFixedSize(32, 32)
        self.reload_btn.clicked.connect(self._reload)
        self.reload_btn.setToolTip("تحديث (F5)")

        self.home_btn = QPushButton("🏠")
        self.home_btn.setFixedSize(32, 32)
        self.home_btn.clicked.connect(self._go_home)

        # شريط العنوان
        self.url_bar = QLineEdit()
        self.url_bar.setPlaceholderText("🔍 أدخل رابطاً أو ابحث...")
        self.url_bar.returnPressed.connect(self._navigate_or_search)

        # مؤشر الأمان
        self.security_label = QLabel("🔒")
        self.security_label.setFixedWidth(25)

        # زر الإشارة المرجعية
        self.bookmark_btn = QPushButton("⭐")
        self.bookmark_btn.setFixedSize(32, 32)
        self.bookmark_btn.clicked.connect(self._add_bookmark)
        self.bookmark_btn.setToolTip("إضافة إشارة مرجعية")

        # زر القائمة الرئيسية
        self.menu_btn = QPushButton("☰")
        self.menu_btn.setFixedSize(32, 32)
        self.menu_btn.clicked.connect(self._show_main_menu)

        # مؤشر مانع الإعلانات
        self.ad_label = QLabel("🛡️0")
        self.ad_label.setToolTip("الإعلانات المحجوبة")

        # إضافة للشريط
        toolbar.addWidget(self.back_btn)
        toolbar.addWidget(self.forward_btn)
        toolbar.addWidget(self.reload_btn)
        toolbar.addWidget(self.home_btn)
        toolbar.addWidget(self.security_label)
        toolbar.addWidget(self.url_bar)
        toolbar.addWidget(self.bookmark_btn)
        toolbar.addWidget(self.ad_label)
        toolbar.addWidget(self.menu_btn)

        # شريط التبويبات
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # زر تبويب جديد
        new_tab_btn = QPushButton("+")
        new_tab_btn.setFixedSize(28, 28)
        new_tab_btn.clicked.connect(lambda: self.add_new_tab())
        self.tabs.setCornerWidget(new_tab_btn, Qt.Corner.TopRightCorner)

        self.setCentralWidget(self.tabs)

        # شريط الحالة
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(14)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)

        # تحديث مؤشر الإعلانات
        self.ad_timer = QTimer()
        self.ad_timer.timeout.connect(self._update_ad_count)
        self.ad_timer.start(2000)

    def _setup_shortcuts(self):
        shortcuts = [
            (QKeySequence("Ctrl+T"), lambda: self.add_new_tab()),
            (QKeySequence("Ctrl+W"), self._close_current_tab),
            (QKeySequence("Ctrl+L"), lambda: self.url_bar.selectAll() or self.url_bar.setFocus()),
            (QKeySequence("F5"), self._reload),
            (QKeySequence("Ctrl+R"), self._reload),
            (QKeySequence("Alt+Left"), self._go_back),
            (QKeySequence("Alt+Right"), self._go_forward),
            (QKeySequence("Ctrl+H"), self._show_history),
            (QKeySequence("Ctrl+D"), self._add_bookmark),
            (QKeySequence("Ctrl+B"), self._show_bookmarks),
            (QKeySequence("Ctrl+J"), self._show_downloads),
            (QKeySequence("Ctrl+Shift+Delete"), self._clear_data),
            (QKeySequence("F11"), self._toggle_fullscreen),
            (QKeySequence("Ctrl+Tab"), self._next_tab),
            (QKeySequence("Ctrl+Shift+Tab"), self._prev_tab),
            (QKeySequence("Ctrl+1"), lambda: self.tabs.setCurrentIndex(0)),
            (QKeySequence("Ctrl+2"), lambda: self.tabs.setCurrentIndex(1)),
            (QKeySequence("Ctrl+3"), lambda: self.tabs.setCurrentIndex(2)),
            (QKeySequence("Escape"), self._stop_loading),
            (QKeySequence("Ctrl+="), self._zoom_in),
            (QKeySequence("Ctrl+-"), self._zoom_out),
            (QKeySequence("Ctrl+0"), self._zoom_reset),
            (QKeySequence("Ctrl+F"), self._find_in_page),
            (QKeySequence("Ctrl+P"), self._print_page),
        ]
        for key, func in shortcuts:
            sc = QShortcut(key, self)
            sc.activated.connect(func)

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
                color: #e0e0e0;
            }
            QToolBar {
                background: #16213e;
                border-bottom: 1px solid #0f3460;
                padding: 4px;
                spacing: 4px;
            }
            QPushButton {
                background: #0f3460;
                color: #e0e0e0;
                border: 1px solid #1a4a7a;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1a4a7a;
                border-color: #4fc3f7;
            }
            QPushButton:pressed {
                background: #0d2d4a;
            }
            QLineEdit {
                background: #0d1b2a;
                color: #e0e0e0;
                border: 1px solid #0f3460;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                selection-background-color: #1a4a7a;
            }
            QLineEdit:focus {
                border-color: #4fc3f7;
            }
            QTabWidget::pane {
                border: none;
                background: #1a1a2e;
            }
            QTabBar::tab {
                background: #16213e;
                color: #aaa;
                border: none;
                padding: 8px 16px;
                min-width: 100px;
                max-width: 200px;
                border-right: 1px solid #0f3460;
            }
            QTabBar::tab:selected {
                background: #0f3460;
                color: #4fc3f7;
                border-bottom: 2px solid #4fc3f7;
            }
            QTabBar::tab:hover {
                background: #1a4a7a;
                color: #e0e0e0;
            }
            QTabBar::close-button {
                subcontrol-position: right;
            }
            QStatusBar {
                background: #16213e;
                color: #888;
                border-top: 1px solid #0f3460;
            }
            QProgressBar {
                background: #0d1b2a;
                border: 1px solid #0f3460;
                border-radius: 4px;
                text-align: center;
                color: #4fc3f7;
            }
            QProgressBar::chunk {
                background: #4fc3f7;
                border-radius: 3px;
            }
            QLabel {
                color: #aaa;
            }
            QMenu {
                background: #16213e;
                color: #e0e0e0;
                border: 1px solid #0f3460;
                border-radius: 6px;
            }
            QMenu::item:selected {
                background: #0f3460;
                color: #4fc3f7;
            }
            QDialog {
                background: #1a1a2e;
                color: #e0e0e0;
            }
            QListWidget {
                background: #16213e;
                color: #e0e0e0;
                border: 1px solid #0f3460;
                border-radius: 6px;
            }
            QListWidget::item:selected {
                background: #0f3460;
                color: #4fc3f7;
            }
            QGroupBox {
                color: #4fc3f7;
                border: 1px solid #0f3460;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QComboBox {
                background: #0d1b2a;
                color: #e0e0e0;
                border: 1px solid #0f3460;
                border-radius: 4px;
                padding: 4px;
            }
        """)

    # ===== إدارة التبويبات =====
    def add_new_tab(self, url=None):
        tab = BrowserTab(self.profile, self.extensions, self)
        tab.title_changed.connect(lambda t, tab=tab: self._update_tab_title(tab, t))
        tab.url_changed.connect(self._on_url_changed)
        tab.loading_changed.connect(self._on_loading_changed)
        tab.view.loadProgress.connect(self._on_progress)

        idx = self.tabs.addTab(tab, "تبويب جديد")
        self.tabs.setCurrentIndex(idx)

        if url:
            tab.navigate(url)
        else:
            tab.navigate(self.settings_data['home'])

        return tab.view

    def _update_tab_title(self, tab, title):
        idx = self.tabs.indexOf(tab)
        if idx >= 0:
            short_title = title[:20] + "..." if len(title) > 20 else title
            self.tabs.setTabText(idx, short_title)
            if idx == self.tabs.currentIndex():
                self.setWindowTitle(f"{title} - المتصفح المطور")

    def _close_tab(self, idx):
        if self.tabs.count() <= 1:
            self.add_new_tab()
        self.tabs.removeTab(idx)

    def _close_current_tab(self):
        self._close_tab(self.tabs.currentIndex())

    def _on_tab_changed(self, idx):
        tab = self.tabs.currentWidget()
        if tab:
            url = tab.view.url().toString()
            self.url_bar.setText(url)
            self._update_security(url)

    def _next_tab(self):
        idx = (self.tabs.currentIndex() + 1) % self.tabs.count()
        self.tabs.setCurrentIndex(idx)

    def _prev_tab(self):
        idx = (self.tabs.currentIndex() - 1) % self.tabs.count()
        self.tabs.setCurrentIndex(idx)

    def _current_view(self):
        tab = self.tabs.currentWidget()
        if tab:
            return tab.view
        return None

    # ===== التنقل =====
    def _navigate_or_search(self):
        text = self.url_bar.text().strip()
        if not text:
            return
        tab = self.tabs.currentWidget()
        if not tab:
            return
        if '.' in text and ' ' not in text:
            tab.navigate(text)
        else:
            tab.search(text, self.settings_data.get('search_engine', 'Google'))

    def _go_back(self):
        view = self._current_view()
        if view:
            view.back()

    def _go_forward(self):
        view = self._current_view()
        if view:
            view.forward()

    def _reload(self):
        view = self._current_view()
        if view:
            view.reload()

    def _go_home(self):
        tab = self.tabs.currentWidget()
        if tab:
            tab.navigate(self.settings_data['home'])

    def _stop_loading(self):
        view = self._current_view()
        if view:
            view.stop()

    # ===== أحداث =====
    def _on_url_changed(self, url):
        if self.tabs.currentWidget() and \
           self.tabs.currentWidget().view.url().toString() == url:
            self.url_bar.setText(url)
            self._update_security(url)
            self._add_to_history(url)

    def _update_security(self, url):
        if url.startswith('https://'):
            self.security_label.setText("🔒")
            self.security_label.setToolTip("اتصال آمن")
        elif url.startswith('http://'):
            self.security_label.setText("⚠️")
            self.security_label.setToolTip("اتصال غير آمن")
        else:
            self.security_label.setText("🌐")

    def _on_loading_changed(self, loading):
        if loading:
            self.reload_btn.setText("✕")
            self.reload_btn.clicked.disconnect()
            self.reload_btn.clicked.connect(self._stop_loading)
            self.progress_bar.show()
        else:
            self.reload_btn.setText("↻")
            self.reload_btn.clicked.disconnect()
            self.reload_btn.clicked.connect(self._reload)
            self.progress_bar.hide()

    def _on_progress(self, progress):
        self.progress_bar.setValue(progress)
        self.status_bar.showMessage(f"جار التحميل... {progress}%")
        if progress == 100:
            self.status_bar.showMessage("تم التحميل", 3000)

    def _update_ad_count(self):
        self.ad_label.setText(f"🛡️{self.ad_blocker.blocked_count}")

    # ===== التاريخ =====
    def _add_to_history(self, url):
        if url in ('', 'about:blank'):
            return
        view = self._current_view()
        title = view.title() if view else url
        entry = {
            'url': url,
            'title': title,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        # تجنب التكرار
        if not self.history or self.history[-1]['url'] != url:
            self.history.append(entry)
            if len(self.history) > 1000:
                self.history.pop(0)

    def _show_history(self):
        dlg = HistoryDialog(self.history, self)
        dlg.navigate_to.connect(self._navigate_to_url)
        dlg.exec()

    def _navigate_to_url(self, url):
        tab = self.tabs.currentWidget()
        if tab:
            tab.navigate(url)

    # ===== الإشارات المرجعية =====
    def _add_bookmark(self):
        view = self._current_view()
        if not view:
            return
        url = view.url().toString()
        title = view.title() or url

        for bm in self.bookmarks:
            if bm['url'] == url:
                QMessageBox.information(self, "موجود", "هذه الصفحة مضافة مسبقاً!")
                return

        self.bookmarks.append({'url': url, 'title': title})
        self.status_bar.showMessage(f"⭐ تمت إضافة '{title}'", 3000)

    def _show_bookmarks(self):
        dlg = BookmarksDialog(self.bookmarks, self)
        dlg.navigate_to.connect(self._navigate_to_url)
        dlg.exec()

    # ===== التنزيلات =====
    def _handle_download(self, download: QWebEngineDownloadRequest):
        default_path = str(Path.home() / "Downloads" / download.suggestedFileName())
        path, _ = QFileDialog.getSaveFileName(
            self, "حفظ الملف", default_path
        )
        if path:
            download.setDownloadFileName(path)
            download.accept()
            dl_info = {
                'name': download.suggestedFileName(),
                'path': path,
                'done': False
            }
            self.downloads.append(dl_info)
            download.isFinishedChanged.connect(
                lambda: dl_info.update({'done': True}) or
                self.status_bar.showMessage(f"✅ تم تنزيل {dl_info['name']}", 5000)
            )
            self.status_bar.showMessage(f"⏳ جار تنزيل {download.suggestedFileName()}...")
        else:
            download.cancel()

    def _show_downloads(self):
        dlg = DownloadsDialog(self.downloads, self)
        dlg.exec()

    # ===== الزووم =====
    def _zoom_in(self):
        view = self._current_view()
        if view:
            view.setZoomFactor(min(view.zoomFactor() + 0.1, 3.0))

    def _zoom_out(self):
        view = self._current_view()
        if view:
            view.setZoomFactor(max(view.zoomFactor() - 0.1, 0.3))

    def _zoom_reset(self):
        view = self._current_view()
        if view:
            view.setZoomFactor(1.0)

    # ===== بحث في الصفحة =====
    def _find_in_page(self):
        text, ok = QInputDialog.getText(self, "بحث في الصفحة", "ابحث عن:")
        if ok and text:
            view = self._current_view()
            if view:
                view.findText(text)

    # ===== طباعة =====
    def _print_page(self):
        view = self._current_view()
        if view:
            view.page().printToPdf(str(Path.home() / "Downloads" / "page.pdf"))
            self.status_bar.showMessage("📄 تم حفظ الصفحة كـ PDF", 3000)

    # ===== وضع ملء الشاشة =====
    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ===== مسح البيانات =====
    def _clear_data(self):
        reply = QMessageBox.question(
            self, "مسح البيانات",
            "هل تريد مسح التاريخ والكوكيز والكاش؟",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.history.clear()
            self.profile.clearAllVisitedLinks()
            self.profile.clearHttpCache()
            self.status_bar.showMessage("✅ تم مسح جميع البيانات", 3000)

    # ===== القائمة الرئيسية =====
    def _show_main_menu(self):
        menu = QMenu(self)

        menu.addAction("📅 التاريخ (Ctrl+H)", self._show_history)
        menu.addAction("⭐ الإشارات المرجعية (Ctrl+B)", self._show_bookmarks)
        menu.addAction("📥 التنزيلات (Ctrl+J)", self._show_downloads)
        menu.addAction("🧩 الملحقات", self._show_extensions)
        menu.addSeparator()
        menu.addAction("⚙️ الإعدادات", self._show_settings)
        menu.addAction("🗑️ مسح البيانات", self._clear_data)
        menu.addSeparator()
        menu.addAction("🖨️ طباعة (Ctrl+P)", self._print_page)
        menu.addAction("🔍 بحث في الصفحة (Ctrl+F)", self._find_in_page)
        menu.addSeparator()
        menu.addAction("ℹ️ عن المتصفح", self._show_about)
        menu.addAction("❌ إغلاق", self.close)

        btn_pos = self.menu_btn.mapToGlobal(QPoint(0, self.menu_btn.height()))
        menu.exec(btn_pos)

    def _show_extensions(self):
        dlg = ExtensionsDialog(self.extensions, self)
        dlg.exec()

    def _show_settings(self):
        dlg = SettingsDialog(self.settings_data, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # تطبيق الإعدادات
            self.ad_blocker.enabled = self.settings_data.get('ad_block', True)
            settings = self.profile.settings()
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.JavascriptEnabled,
                self.settings_data.get('javascript', True)
            )

    def _show_about(self):
        QMessageBox.about(self, "عن المتصفح",
            "🌐 المتصفح المطور\n\n"
            "متصفح مفتوح المصدر مبني بـ Python + PyQt6\n"
            "يعمل على Arch Linux\n\n"
            "الميزات:\n"
            "• مانع إعلانات مدمج\n"
            "• تعدد التبويبات\n"
            "• إشارات مرجعية وتاريخ\n"
            "• تحميل الملحقات\n"
            "• وضع ملء الشاشة\n"
            "• تحميل الملفات\n"
        )

    # ===== حفظ وتحميل البيانات =====
    def _load_data(self):
        data_file = self.data_dir / "browser_data.json"
        if data_file.exists():
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
                    self.bookmarks = data.get('bookmarks', [])
                    self.extensions = data.get('extensions', [])
                    self.settings_data.update(data.get('settings', {}))
            except Exception:
                pass

    def _save_data(self):
        data_file = self.data_dir / "browser_data.json"
        try:
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'history': self.history[-500:],
                    'bookmarks': self.bookmarks,
                    'extensions': self.extensions,
                    'settings': self.settings_data
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def closeEvent(self, event):
        self._save_data()
        event.accept()


# ===== نقطة الدخول =====
def main():
    # إعدادات Wayland/X11
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--no-sandbox --disable-gpu-sandbox')
    
    app = QApplication(sys.argv)
    app.setApplicationName("المتصفح المطور")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("OpenBrowser")

    # خط مناسب للعربية
    font = QFont("Noto Sans Arabic", 10)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(font)

    window = MainBrowser()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
