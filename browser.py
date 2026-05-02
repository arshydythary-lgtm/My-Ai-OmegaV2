#!/usr/bin/env python3
"""
متصفح سريع وخفيف - إصدار محسّن
الأولوية: السرعة، الأداء، استهلاك رام منخفض
"""

import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLineEdit, QPushButton, QToolBar, QStatusBar,
    QMenu, QDialog, QListWidget, QLabel, QFileDialog, QMessageBox,
    QProgressBar, QCheckBox, QSlider, QComboBox, QTextEdit, 
    QGroupBox, QScrollArea, QListWidgetItem, QInputDialog,
    QColorDialog, QFormLayout, QSpinBox
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import (
    QWebEngineProfile, QWebEnginePage, QWebEngineUrlRequestInterceptor,
    QWebEngineDownloadRequest, QWebEngineSettings
)
from PyQt6.QtCore import QUrl, Qt, QTimer, QSize, QPoint, pyqtSignal
from PyQt6.QtGui import (
    QFont, QAction, QKeySequence, QShortcut, QColor
)

# ===== مانع الإعلانات خفيف =====
AD_DOMAINS = {
    "doubleclick.net", "googlesyndication.com", "googleadservices.com",
    "google-analytics.com", "googletagmanager.com", "facebook.com/tr",
    "ads.youtube.com", "amazon-adsystem.com", "taboola.com", "outbrain.com"
}

class AdBlocker(QWebEngineUrlRequestInterceptor):
    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.blocked_count = 0

    def interceptRequest(self, info):
        if not self.enabled:
            return
        host = info.requestUrl().host()
        url = info.requestUrl().toString()
        
        # فحص سريع للمجالات المحظورة
        for domain in AD_DOMAINS:
            if domain in host:
                info.block(True)
                self.blocked_count += 1
                return
        
        # فحص أنماط URL
        if any(p in url for p in ['/ads/', '/ad/', '/tracking/', '/analytics/']):
            info.block(True)
            self.blocked_count += 1


# ===== صفحة الويب المخصصة =====
class BrowserPage(QWebEnginePage):
    def __init__(self, profile, parent=None):
        super().__init__(profile, parent)

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceId):
        pass

    def createWindow(self, type_):
        if hasattr(self.parent(), 'create_new_tab'):
            return self.parent().create_new_tab().page()
        return super().createWindow(type_)


# ===== عارض الويب المخصص =====
class BrowserView(QWebEngineView):
    def __init__(self, profile, parent=None):
        super().__init__(parent)
        self.tab_widget = parent
        page = BrowserPage(profile, self)
        self.setPage(page)
        # إزالة المستطيل الأزرق عند التحديد
        self.setStyleSheet("QWebEngineView { outline: none; }")

    def create_new_tab(self):
        if self.tab_widget and hasattr(self.tab_widget, 'add_new_tab'):
            return self.tab_widget.add_new_tab()
        return None

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        menu.addSeparator()
        
        save_action = QAction("💾 حفظ", self)
        save_action.triggered.connect(lambda: self.page().save(str(Path.home() / "Downloads" / "page.html")))
        menu.addAction(save_action)
        
        menu.exec(event.globalPos())


# ===== نوافذ بسيطة =====
class SimpleListDialog(QDialog):
    """نافذة قائمة بسيطة للتاريخ والإشارات"""
    navigate_to = pyqtSignal(str)

    def __init__(self, title, items, parent=None, show_delete=False, show_clear=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(600, 400)
        self.items = items
        self.show_delete = show_delete
        self.show_clear = show_clear
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        if self.show_delete:
            del_btn = QPushButton("🗑️ حذف")
            del_btn.clicked.connect(self._delete_selected)
            btn_layout.addWidget(del_btn)
        if self.show_clear:
            clear_btn = QPushButton("🗑️ مسح الكل")
            clear_btn.clicked.connect(self._clear_all)
            btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        close_btn = QPushButton("❌ إغلاق")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        self._populate()

    def _populate(self):
        self.list_widget.clear()
        for item in reversed(self.items) if hasattr(self, 'items') else self.items:
            if isinstance(item, dict):
                text = f"{item.get('title', item.get('url', ''))}\n{item['url']}"
                self.list_widget.addItem(text)
            else:
                self.list_widget.addItem(str(item))

    def _on_item_clicked(self, item):
        row = self.list_widget.row(item)
        data = self.items[row] if row < len(self.items) else None
        if data and isinstance(data, dict) and 'url' in data:
            self.navigate_to.emit(data['url'])
        self.close()

    def _delete_selected(self):
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.items):
            self.items.pop(row)
            self._populate()

    def _clear_all(self):
        self.items.clear()
        self._populate()


# ===== نافذة الإعدادات الكاملة =====
class SettingsDialog(QDialog):
    def __init__(self, settings_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ الإعدادات")
        self.resize(550, 500)
        self.settings_data = settings_data
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        container = QWidget()
        container_layout = QVBoxLayout(container)
        
        # الصفحة الرئيسية
        home_group = QGroupBox("🏠 الصفحة الرئيسية")
        home_layout = QHBoxLayout()
        self.home_edit = QLineEdit(self.settings_data.get('home', 'https://google.com'))
        home_layout.addWidget(QLabel("الرابط:"))
        home_layout.addWidget(self.home_edit)
        home_group.setLayout(home_layout)
        container_layout.addWidget(home_group)

        # محرك البحث
        search_group = QGroupBox("🔍 محرك البحث")
        search_layout = QHBoxLayout()
        self.search_combo = QComboBox()
        self.search_combo.addItems(["Google", "DuckDuckGo", "Bing", "Brave"])
        self.search_combo.setCurrentText(self.settings_data.get('search_engine', 'Google'))
        search_layout.addWidget(QLabel("المحرك:"))
        search_layout.addWidget(self.search_combo)
        search_group.setLayout(search_layout)
        container_layout.addWidget(search_group)

        # الأداء و GPU
        perf_group = QGroupBox("⚡ الأداء و GPU")
        perf_layout = QVBoxLayout()
        self.gpu_cb = QCheckBox("تسريع GPU (يحتاج إعادة تشغيل)")
        self.gpu_cb.setChecked(self.settings_data.get('gpu_acceleration', True))
        self.cache_limit = QSpinBox()
        self.cache_limit.setRange(64, 2048)
        self.cache_limit.setValue(self.settings_data.get('cache_limit_mb', 256))
        self.cache_limit.setSuffix(" MB")
        perf_layout.addWidget(self.gpu_cb)
        perf_layout.addWidget(QLabel("حد الكاش:"))
        perf_layout.addWidget(self.cache_limit)
        perf_group.setLayout(perf_layout)
        container_layout.addWidget(perf_group)

        # الخصوصية
        privacy_group = QGroupBox("🔒 الخصوصية والحماية")
        privacy_layout = QVBoxLayout()
        self.ad_block_cb = QCheckBox("مانع الإعلانات")
        self.ad_block_cb.setChecked(self.settings_data.get('ad_block', True))
        self.js_cb = QCheckBox("JavaScript")
        self.js_cb.setChecked(self.settings_data.get('javascript', True))
        self.cookies_cb = QCheckBox("الكوكيز")
        self.cookies_cb.setChecked(self.settings_data.get('cookies', True))
        self.images_cb = QCheckBox("الصور")
        self.images_cb.setChecked(self.settings_data.get('images', True))
        privacy_layout.addWidget(self.ad_block_cb)
        privacy_layout.addWidget(self.js_cb)
        privacy_layout.addWidget(self.cookies_cb)
        privacy_layout.addWidget(self.images_cb)
        privacy_group.setLayout(privacy_layout)
        container_layout.addWidget(privacy_group)

        # الألوان
        colors_group = QGroupBox("🎨 تخصيص الألوان")
        colors_layout = QFormLayout()
        
        self.bg_color_btn = QPushButton("لون الخلفية")
        self.bg_color_btn.setStyleSheet(f"background-color: {self.settings_data.get('bg_color', '#1a1a2e')}")
        self.bg_color_btn.clicked.connect(lambda: self._pick_color('bg'))
        
        self.fg_color_btn = QPushButton("لون النص")
        self.fg_color_btn.setStyleSheet(f"background-color: {self.settings_data.get('fg_color', '#e0e0e0')}")
        self.fg_color_btn.clicked.connect(lambda: self._pick_color('fg'))
        
        self.accent_color_btn = QPushButton("لون التمييز")
        self.accent_color_btn.setStyleSheet(f"background-color: {self.settings_data.get('accent_color', '#4fc3f7')}")
        self.accent_color_btn.clicked.connect(lambda: self._pick_color('accent'))
        
        colors_layout.addRow(self.bg_color_btn)
        colors_layout.addRow(self.fg_color_btn)
        colors_layout.addRow(self.accent_color_btn)
        colors_group.setLayout(colors_layout)
        container_layout.addWidget(colors_group)

        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        # أزرار
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("💾 حفظ")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("❌ إلغاء")
        cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _pick_color(self, color_type):
        color = QColorDialog.getColor()
        if color.isValid():
            hex_color = color.name()
            if color_type == 'bg':
                self.settings_data['bg_color'] = hex_color
                self.bg_color_btn.setStyleSheet(f"background-color: {hex_color}")
            elif color_type == 'fg':
                self.settings_data['fg_color'] = hex_color
                self.fg_color_btn.setStyleSheet(f"background-color: {hex_color}")
            elif color_type == 'accent':
                self.settings_data['accent_color'] = hex_color
                self.accent_color_btn.setStyleSheet(f"background-color: {hex_color}")

    def _save(self):
        self.settings_data['home'] = self.home_edit.text()
        self.settings_data['search_engine'] = self.search_combo.currentText()
        self.settings_data['gpu_acceleration'] = self.gpu_cb.isChecked()
        self.settings_data['cache_limit_mb'] = self.cache_limit.value()
        self.settings_data['ad_block'] = self.ad_block_cb.isChecked()
        self.settings_data['javascript'] = self.js_cb.isChecked()
        self.settings_data['cookies'] = self.cookies_cb.isChecked()
        self.settings_data['images'] = self.images_cb.isChecked()
        self.accept()


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
        widget = self.tabs.widget(idx)
        if widget:
            widget.deleteLater()
        self.tabs.removeTab(idx)
        if self.tabs.count() == 0:
            self.close()
    def _close_tab(self, idx):
        widget = self.tabs.widget(idx)
        if widget:
            widget.deleteLater()
        self.tabs.removeTab(idx)
        if self.tabs.count() == 0:
            self.close()
    def _close_tab(self, idx):
        widget = self.tabs.widget(idx)
        if widget:
            widget.deleteLater()
        self.tabs.removeTab(idx)
        if self.tabs.count() == 0:
            self.close()
    def _close_tab(self, idx):
        widget = self.tabs.widget(idx)
        if widget:
            widget.deleteLater()
        self.tabs.removeTab(idx)
        if self.tabs.count() == 0:
            self.close()

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
