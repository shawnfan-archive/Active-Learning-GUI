import Vue from 'vue'
import App from './App.vue'
import Router from './router.js'
import VueKonva from 'vue-konva'
import vuetify from './plugins/vuetify';
import 'roboto-fontface/css/roboto/roboto-fontface.css'
import '@mdi/font/css/materialdesignicons.css'

Vue.config.productionTip = false

Vue.use(VueKonva)

new Vue({
  router: Router,
  template: "<App/>",
  vuetify,
  render: h => h(App)
}).$mount('#app')
